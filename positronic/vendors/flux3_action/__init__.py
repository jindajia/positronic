"""FLUX 3 Action DROID, served over positronic's session protocol. NOTICE states its licence."""

# The request and reply fields of Black Forest Labs' RoboLab server (`flux_action.serving.robolab`), which
# `backend.py` runs. The server stacks the wrist view above the two exterior views.
JOINT_POSITION = 'observation/joint_position'
GRIPPER_POSITION = 'observation/gripper_position'
WRIST_IMAGE = 'observation/wrist_image_left'
EXTERIOR_IMAGE_1 = 'observation/exterior_image_1_left'
EXTERIOR_IMAGE_2 = 'observation/exterior_image_2_left'
PROMPT = 'prompt'
ACTIONS = 'action'
# A request field that `backend.py` reads and BFL's server never sees: the denoising steps of that chunk.
NUM_INFERENCE_STEPS = 'num_inference_steps'
# A request field that `server.py` reads and removes: what the client says of this call and of the ones before it.
CALL = 'call'
