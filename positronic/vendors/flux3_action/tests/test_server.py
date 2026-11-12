"""The FLUX 3 Action server, run against a stand-in for BFL's interpreter and its RoboLab server."""

import json
import logging
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from positronic_model_server import keys as offboard_keys
from positronic_model_server import server_wire, websocket_wire
from positronic_model_server.protocol import AUTH_HEADER, bearer
from positronic_model_server.spec import ARGS, SEQ
from positronic_wire import websocket, wire
from positronic_wire.roboarena import TextAnswer

from positronic import keys
from positronic.offboard import roboarena
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import PolicyServer
from positronic.policy import keys as policy_keys
from positronic.policy.processors import ChunkedSchedule
from positronic.vendors import flux3_action
from positronic.vendors.flux3_action import codecs, server
from positronic.vendors.flux3_action.tests import bfl_stand_in

TOKEN = 'run-token'
REPOSITORY = 'black-forest-labs/flux-3-action-droid'
REVISION = 'policy-commit'
SUBFOLDER = 'variants/gd'
LOG = 'backend.jsonl'


def _flux_python(directory: Path, mode: bfl_stand_in.Mode) -> str:
    """An interpreter whose run of `backend.py` is the stand-in, answering in `mode`."""
    wrapper = directory / 'python'
    stand_in = bfl_stand_in.__file__
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stand_in}" "{directory / LOG}" {mode.value} "$@"\n')
    wrapper.chmod(0o755)
    return str(wrapper)


def _log(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (directory / LOG).read_text().splitlines()]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _observation() -> dict[str, Any]:
    """What an eval that renders one exterior view sends, as MolmoSpaces does."""
    return {
        keys.WRIST_IMAGE: np.zeros((720, 1280, 3), dtype=np.uint8),
        keys.EXTERIOR_IMAGE: np.zeros((480, 640, 3), dtype=np.uint8),
        keys.JOINTS: np.arange(7, dtype=np.float32),
        keys.GRIP: 0.25,
        keys.TASK: 'put the cube in the bowl',
    }


def _session(
    policy_server: PolicyServer, obs: dict[str, Any], query: str = ''
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The metadata and the commands of one inference of one session on `query`, behind the run's token."""
    wires = [websocket_wire.WebsocketWire(server_wire.ServedHostPort('127.0.0.1', 0))]
    ready = threading.Event()
    thread = threading.Thread(target=policy_server.serve, args=(wires, ready.set), daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=30)
        served = wires[0].served_address
        assert isinstance(served, server_wire.ServedHostPort)
        address = wire.HostPortAddress('127.0.0.1', served.port, wire.SESSION_PATH, query)
        client = InferenceClient(websocket.WebsocketClientWire(), address, headers={AUTH_HEADER: bearer(TOKEN)})
        client.keepalive()
        session = client.new_session()
        try:
            return session.metadata, session.infer(obs)
        finally:
            session.close()
    finally:
        policy_server.shutdown()
        thread.join(timeout=30)
        assert not thread.is_alive()


def test_a_session_reaches_bfls_server_in_its_fields_and_gets_the_whole_chunk(tmp_path):
    port = _free_port()
    model = server.flux3_action_model.override(
        checkpoint=REPOSITORY,
        revision=REVISION,
        subfolder=SUBFOLDER,
        flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.SERVE),
        backend_port=port,
    )

    meta, actions = _session(PolicyServer(model, server.pipeline, auth_token=TOKEN), _observation())

    assert meta[offboard_keys.LOCAL_STACK][SEQ][1][ARGS]['merge']['horizon_sec'] is None
    assert len(actions) == len(bfl_stand_in.CHUNK)
    for action, expected in zip(actions, bfl_stand_in.CHUNK, strict=True):
        np.testing.assert_allclose(action[keys.ROBOT_COMMAND].positions, expected[:7])
        assert action[keys.TARGET_GRIP] == pytest.approx(expected[7])
    # One warm-up request comes before the session's own.
    started, _warm_up, request = _log(tmp_path)
    # rules-allow: hardcoded-keys — BFL's RoboLab fields as its documentation spells them, pinned apart from the
    # package constants so a typo in a constant fails here.
    assert request['request'] == {
        'observation/wrist_image_left': [360, 640, 3],
        'observation/exterior_image_1_left': [360, 640, 3],
        'observation/exterior_image_2_left': [360, 640, 3],
        'observation/joint_position': list(range(7)),
        'observation/gripper_position': [0.25],
        'prompt': 'put the cube in the bowl',
        roboarena.ENDPOINT: roboarena.INFER,
    }
    assert started['argv'][:2] == ['-P', str(server.BACKEND_SCRIPT)]
    assert dict(zip(started['argv'][2::2], started['argv'][3::2], strict=True)) == {
        '--checkpoint': REPOSITORY,
        '--port': str(port),
        '--revision': REVISION,
        '--subfolder': SUBFOLDER,
    }
    assert not _alive(started['pid']), 'the server left the backend running after it stopped'


def test_a_session_that_names_a_horizon_tells_the_client_to_play_that_much_of_each_chunk(tmp_path):
    model = server.flux3_action_model.override(
        checkpoint=str(tmp_path), flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.SERVE), backend_port=_free_port()
    )

    meta, actions = _session(
        PolicyServer(model, server.pipeline, auth_token=TOKEN), _observation(), query='horizon_sec=0.4'
    )

    assert meta[offboard_keys.LOCAL_STACK][SEQ][1][ARGS]['merge']['horizon_sec'] == 0.4
    # The server sends the whole chunk, and the client stack discards the commands past the horizon.
    assert len(actions) == len(bfl_stand_in.CHUNK)


def test_a_local_policy_directory_reaches_the_backend_with_no_revision_and_no_subfolder(tmp_path):
    model = server.flux3_action_model(
        checkpoint=str(tmp_path), flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.SERVE), backend_port=_free_port()
    )
    model.close()
    argv = _log(tmp_path)[0]['argv']
    assert dict(zip(argv[2::2], argv[3::2], strict=True)).keys() == {'--checkpoint', '--port'}


def test_a_session_that_names_a_denoising_step_count_sends_it_with_each_request(tmp_path):
    model = server.flux3_action_model.override(
        checkpoint=str(tmp_path), flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.SERVE), backend_port=_free_port()
    )

    meta, actions = _session(
        PolicyServer(model, server.pipeline, auth_token=TOKEN), _observation(), query='num_inference_steps=2'
    )

    assert meta[codecs.NUM_INFERENCE_STEPS_META] == 2
    assert _log(tmp_path)[-1]['request'][flux3_action.NUM_INFERENCE_STEPS] == 2
    assert len(actions) == len(bfl_stand_in.CHUNK)


def test_the_server_logs_each_request_with_what_the_client_says_of_the_call(tmp_path, caplog):
    model = server.flux3_action_model.override(
        checkpoint=str(tmp_path), flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.SERVE), backend_port=_free_port()
    )
    earlier = [{policy_keys.CALL_ID: 2, policy_keys.ELAPSED_SEC: 0.5}]
    obs = {**_observation(), policy_keys.CALL_ID: 3, policy_keys.CALL_TIMINGS: earlier}

    with caplog.at_level(logging.INFO, logger=server.__name__):
        meta, _actions = _session(
            PolicyServer(model, server.pipeline, auth_token=TOKEN), obs, query='num_inference_steps=2&horizon_sec=0.5'
        )

    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith(server.REQUEST_LOG)]
    # One warm-up request comes before the session's own, and it states no call.
    warm_up, request = (json.loads(line.removeprefix(server.REQUEST_LOG)) for line in lines)
    assert policy_keys.CALL_ID not in warm_up
    assert request.pop(server.LOG_BACKEND_MS) > 0
    assert request.pop(server.LOG_SESSION)
    assert request == {
        policy_keys.CALL_ID: 3,
        policy_keys.CALL_TIMINGS: earlier,
        ChunkedSchedule.HORIZON_SEC_ARG: 0.5,
        flux3_action.NUM_INFERENCE_STEPS: 2,
    }
    # BFL's server never sees the field.
    assert flux3_action.CALL not in _log(tmp_path)[-1]['request']


@pytest.mark.parametrize('count', ['0', '1.5', '"two"'])
def test_a_session_that_names_no_positive_whole_step_count_is_refused(tmp_path, count):
    model = server.flux3_action_model.override(
        checkpoint=str(tmp_path), flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.SERVE), backend_port=_free_port()
    )

    with pytest.raises(RuntimeError, match='num_inference_steps must be a positive integer'):
        _session(
            PolicyServer(model, server.pipeline, auth_token=TOKEN), _observation(), query=f'num_inference_steps={count}'
        )


def test_a_warm_up_the_backend_refuses_fails_the_load_and_stops_the_backend(tmp_path):
    with pytest.raises(TextAnswer, match='the policy refused the request'):
        server.flux3_action_model(
            checkpoint=REPOSITORY,
            flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.REFUSE),
            backend_port=_free_port(),
        )
    assert not _alive(_log(tmp_path)[0]['pid'])


def test_a_backend_that_exits_fails_the_load(tmp_path):
    exits = tmp_path / 'python'
    exits.write_text('#!/bin/sh\nexit 3\n')
    exits.chmod(0o755)
    with pytest.raises(RuntimeError, match='exited with code 3'):
        server.flux3_action_model(checkpoint=REPOSITORY, flux_python=str(exits), backend_port=_free_port())
