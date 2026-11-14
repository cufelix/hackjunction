# Home-pose camera frames (sim) — are the hands in the head view?

- **What:** head RGB, head depth and both wrist RGB images at the sim HOME pose, the pose the evaluator leaves the arms in
  (`evaluate.home_arms()`, then `env.reset()` with the layout of seed 1000). Stock sim, stock colours, nominal camera.
- **Arm pose:** sim profile `home_arm_q` (config.yaml `profiles.sim`):
  left `[0.0, 0.35, 0.0, -0.5, 0.0, 0.0, 0.0]`, right `[0.0, -0.35, 0.0, -0.5, 0.0, 0.0, 0.0]`.
  Hand joints were not logged by the dump script; they are the sim task's default hand pose at home.
- **Answer:** YES, in the sim both Dex3 hands and forearms are clearly in the head image (left hand down the left
  quarter, right fingertips over the mat at the right edge). So "home => out of view" is false in the sim, and arm
  masks must come from FK.
- **Caveat for the real robot:** the ROBOT profile `home_arm_q` (config.yaml top level, `[0.2, 0.7, 0.4, -0.2, 0, -0.5, 0]`
  mirrored) is documented as keeping the hands OUT of the head view. Hand-in-view features (FK mask is fine either way;
  hand-calib B1 needs the hands in view) behave differently on the real robot unless a calibration pose is used.

Files: `home_frame.png` (head RGB 640x480), `home_depth.png` (head depth, visualised), `home_left_wrist_rgb.png`,
`home_right_wrist_rgb.png`.
