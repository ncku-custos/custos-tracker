# Per-frame single-object-tracker output. Box fields use ctrk's convention:
# top-left origin, pixel units (NOT the vision_msgs center convention).
# IDLE means the node is active but no target has been set (pre-init or after
# reset); box and score are zero and must be ignored.

std_msgs/Header header

uint8 STATE_IDLE=0
uint8 STATE_TRACKING=1
uint8 STATE_UNSTABLE=2
uint8 STATE_LOST=3
uint8 state

float32 x
float32 y
float32 w
float32 h
float32 score
