"""The server-side DROID codecs of FLUX 3 Action. They encode an observation as a request to BFL's server, and
decode its actions as commands."""

from positronic import keys
from positronic.cfg import codecs
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import Codec
from positronic.policy.processors import ChunkedSchedule
from positronic.vendors import flux3_action

# The policy learned from DROID recordings at 15 Hz, and it predicts actions at that rate.
FPS = 15.0
# Each view has this (width, height). BFL's server refuses a wrist view of another size.
VIEW_SIZE = (640, 360)

# The session metadata field that names the denoising step count a session asks for.
NUM_INFERENCE_STEPS_META = 'flux3_action.num_inference_steps'


class DenoisingSteps(Codec):
    """Asks `backend.py` for `count` denoising steps of each chunk."""

    def __init__(self, count: int):
        if type(count) is not int or count < 1:
            raise ValueError(f'num_inference_steps must be a positive integer, got {count!r}')
        self._count = count

    def encode(self, data: dict) -> dict:
        return {flux3_action.NUM_INFERENCE_STEPS: self._count}

    @property
    def meta(self) -> dict:
        return {NUM_INFERENCE_STEPS_META: self._count}


class CallRecord(Codec):
    """Hands `server.py` what the client says of each call, for the server's request log.

    The client's schedule numbers the calls of a session, and sends with each one how long the calls answered
    since the last one took on its own clock. In a simulated trial that charges inference time, that is the time
    the trial paid for the call. `horizon_sec` is the session's own, which no request states.
    """

    def __init__(self, horizon_sec: float | None):
        self._horizon_sec = horizon_sec

    def encode(self, data: dict) -> dict:
        return {
            flux3_action.CALL: {
                policy_keys.CALL_ID: data.get(policy_keys.CALL_ID),
                policy_keys.CALL_TIMINGS: [dict(timing) for timing in data.get(policy_keys.CALL_TIMINGS, ())],
                ChunkedSchedule.HORIZON_SEC_ARG: self._horizon_sec,
            }
        }


_views_3cam = {
    flux3_action.WRIST_IMAGE: (keys.WRIST_IMAGE, VIEW_SIZE),
    flux3_action.EXTERIOR_IMAGE_1: (keys.EXTERIOR_IMAGE, VIEW_SIZE),
    flux3_action.EXTERIOR_IMAGE_2: (keys.EXTERIOR_IMAGE_2, VIEW_SIZE),
}
_obs_3cam = codecs.observation.override(
    state={flux3_action.JOINT_POSITION: {keys.JOINTS: 7}, flux3_action.GRIPPER_POSITION: {keys.GRIP: 1}},
    images=_views_3cam,
    task_field=flux3_action.PROMPT,
)
# Each action holds seven absolute joint positions and the gripper's closed fraction, as DROID records them.
_action = codecs.droid_execution.override(
    action=codecs.absolute_joints_action.override(tgt_joints_key=keys.JOINTS, tgt_grip_key=keys.GRIP)
)

# This codec sends both exterior views, as the policy learned from them.
droid_3cam = codecs.compose.override(obs=_obs_3cam, action=_action, training_fps=FPS)
# This codec sends one exterior view in both exterior slots, for an eval that renders only one.
droid = droid_3cam.override(
    obs=_obs_3cam.override(images={**_views_3cam, flux3_action.EXTERIOR_IMAGE_2: (keys.EXTERIOR_IMAGE, VIEW_SIZE)})
)
