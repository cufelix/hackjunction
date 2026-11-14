# Real-frame recall audit: `finals_r21` and `rc1`

Scope: the stock classical RGB/RGB-D path in this branch, using the real D435i head-camera calibration. The real
audit frames have no depth, so the plane fallback is the operative localisation path. No code under
`aaltonomous/interface/` or `deploy/` is changed by the recall work.

## Effective configurations

Both overlays use `perception.min_area_px: 120`, `open_px: 3`, `erode_px: 2`, and touching-blob splitting. Both
classify shape, but shape classification normally changes the shape label rather than removing the colour candidate.

| Setting | `finals_r21` | `rc1` |
|---|---:|---:|
| HSV set | `robust_v2` | `robust` |
| red/orange band | H 0–22 or 170–180, S>=70, V>=70 | H 0–16 or 170–180, S>=70, V>=70 |
| yellow band | H 24–60, S>=35, V>=90 | H 18–52, S>=45, V>=90 |
| blue/purple band | H 100–145, S>=70, V>=50 | H 100–145, S>=70, V>=50 |
| hue guard | on; H 22–24 abstain, cross-label gap <6 H units abstains, red top S<110 abstains | off |
| colour gate | on | off |
| saturation post-floor | purple/blue median S>=110; yellow median S>=45 | off |
| distance area floor/ceiling | 0.4x / 3.0x projected 45 mm square | off |
| workspace gate | pelvis x 0.12–min(0.45, bin near edge-0.01), abs(y)<=0.40 | off |
| FK arm/hand mask | on, configured dilation 0 px | off |
| online counter/pitch work | counter height on; table pitch calibration on | off |

The measured real values in the problem statement are inside the hue bands: orange H~12, yellow H~38–40, and
purple H~111. Purple is nevertheless clipped twice: pixels below S=70 never enter its HSV mask, then the finals
post-gate rejects a surviving purple component whose median S is below 110. Yellow S~76–77 and orange S~180–200
comfortably clear their relevant saturation limits.

## RGB frame to policy candidate list

1. `ClassicalPolicy._act()` reaches `look` only after the commanded home trajectory and a `motion.settle_s` hold.
   In `finals_r21`, `policy.safety_shell` runs `_look()` in a low-priority worker while `act()` returns bounded hold
   actions. `rc1` does this only if its caller explicitly constructed a threaded policy.
2. `_look()` calls `Perception.detect(head_rgb, head_depth, robot_q=...)`. The finals FK argument is present when
   measured joints are available; rc1 does not request it.
3. `detect()` updates the optional runtime counter height, then builds `_keep_mask()`.
   - With usable depth, each valid pixel is deprojected and retained only at 0.25–2.0 object heights over the mat,
     outside bin footprints plus 4 mm.
   - On the audited no-depth frames, the mask is 255 except for projected 3-D bin hulls. Pixels in those hulls are
     gone before components are formed.
   - Finals additionally removes the projected FK arm/hand surface. A real object behind the arm is therefore
     shortened or split before its area is measured.
4. RGB is converted to OpenCV HSV. `_masks_hsv()` unions each class's threshold bands and applies a 3x3
   morphological opening. The mask is intersected with the keep/FK mask and opened with 3x3 **again**. Thin tips,
   small triangular regions, disconnected specular regions, and one/two-pixel bridges can therefore disappear.
5. Finals hue-guard grouping unions the red and yellow masks before connected components. It labels each union
   component from the median HSV of its brightest 35% of pixels by V. A hue in the 22–24 gap falls back to the
   majority mask initially, then is rejected by the guard. rc1 groups every colour independently and does no
   top-face hue vote.
6. Eight-connected components are computed. Any component below 120 px is silently dropped. This is the first
   unconditional blob-area floor and applies to both overlays.
7. With `split_blobs` on, a blob is considered for splitting above 1.5 times the expected 45 mm-square area
   (`1.2*(fx*0.045/Z)^2`). Distance-transform parts must each exceed 0.3 times that expected area; the fallback
   major-axis split requires aspect ratio >=1.4. A merged blob that cannot meet these part rules remains one blob.
8. `_detect_blob()` erodes by two pixels only for depth sampling. It restores the original blob if the eroded core
   is below 25% of the component area. With no depth, fewer than 25 valid depth pixels selects `_from_plane()`:
   the component centroid ray is intersected at `counter_z + object_height - 0.25*object_height`. A ray that does
   not point down produces non-finite coordinates and is rejected.
9. `_reject()` drops non-finite 3-D positions, heights outside 0.4–2.0 object heights, centres outside the mat plus
   2 cm, and centres inside any bin footprint plus 1 cm.
10. Finals runs the colour post-gate:
    - purple median S<110 or yellow median S<45 is rejected;
    - with projected depth Z, area must be between 0.4 and 3.0 times
      `fx*fy*(0.045/Z)^2`; without a projectable point it must be 1,000–6,000 px;
    - the positive workspace and bin-footprint/bin-edge rules above must pass.
    rc1 skips this entire stage.
11. The returned detections enter `_choose()`. Failed/attempt-exhausted tracks, missing instruction mappings,
    placement-confidence gates (off in these overlays), time limits, and failed motion plans can remove a detection
    from the **pickable policy list** even though perception found it. In finals, retry association is 6 cm and
    permits up to three varied attempts; rc1 has the same retry settings. A perception-empty or wholly filtered
    list currently becomes `nothing left to sort` immediately.

## Projected object areas at `counter_z=0.15`, zero pitch correction

The real calibration is 640x480 with `fx=604.827`, `fy=604.298`, and camera origin z=0.47387 m. Object height is
45 mm, so the horizontal top face is at z=0.195 m. “Pitch 0” below means no online correction applied to the
calibrated mount transform. For a horizontal patch of world area A at optical depth Z, the local pinhole area is:

`area_px = fx * fy * abs(z_top - z_camera) * A / Z^3`.

The object region spans pelvis x about 0.26–0.35 m. At y=0 this maps from image row 313 (near) through 255
(middle) to 205 (far). The ideal, unoccluded top-face areas are:

| pelvis x / row / optical Z | cube, A=s^2 | cylinder, A=pi(s/2)^2 | prism, A=s^2/sqrt(3) |
|---|---:|---:|---:|
| 0.260 m / row 313 / 0.345 m | 5,023 px | 3,945 px | 2,900 px |
| 0.305 m / row 255 / 0.375 m | 3,901 px | 3,064 px | 2,252 px |
| 0.350 m / row 205 / 0.406 m | 3,090 px | 2,427 px | 1,784 px |

The current finals area model ignores both foreshortening and shape area. Its 0.4x floors at those rows are
2,486, 2,101, and 1,798 px. Thus a complete far-row prism top (1,784 px) is already below the 1,798 px floor;
near/middle prisms have only 414/151 px of margin before opening, glare, perspective side loss, or occlusion.
The 3.0x ceilings are 18,647, 15,755, and 13,487 px. A single object will not hit them, but an unsplit same-colour
merge, bin/tape region, or arm-attached region can.

The proposed geometry gate uses 0.35 times the projected prism area for the floor and 2.0 times the projected cube
area for the ceiling. Across the same rows those bounds are 1,015–10,047 px, 788–7,802 px, and 624–6,179 px. It
therefore allows a substantially occluded prism while still rejecting large merged/bin regions.

## Why each weak class is lost

### Purple cube (recall 0.50)

- H~111 is correct in both overlays, consistent with 0.98 colour accuracy once found.
- Real top/washed pixels with S 60–69 are removed by the HSV S>=70 threshold before grouping. Surviving S 70–109
  pixels form a plausible component, but finals rejects it when the whole-blob median is below 110. A true object
  spanning S 60–110 can therefore lose more than half its pixels at segmentation and then fail the post-floor.
- rc1 has no S=110 post-floor, so its remaining purple-specific loss is the HSV S=70 threshold, double opening,
  120 px component floor, and common geometry rejects.

### Orange prism (recall 0.38)

- H~12 and S~180–200 are strong; colour itself is not the cause.
- Its ideal top area is only 57.7% of the cube's. Finals compares it with a square-area model, placing the full far
  top below the floor and leaving almost no margin elsewhere. A partial prism cut by an arm/hand or neighbour fails
  even earlier.
- Two 3x3 openings preferentially erase a triangle's narrow apex. The remaining blob can fall below 120 px or the
  finals geometric floor.
- The brightest-35% hue subset contains only about 35% of an already small/fragmented mask. A bright side, glare,
  or neighbour bridge can dominate it and move the component into the hue abstain/gap decision.
- There is no fill-ratio or convexity rejection of the **colour candidate**. Shape classification does compute
  convex-hull/min-area-rectangle fill with cube>=0.88, cylinder>=0.66, triangle below 0.66 (blob fallback requires
  60 px), but it only assigns/abstains the shape label. `prism_pose` fill 0.70–0.85 and apex checks affect grasp
  posture/confidence, not whether the coloured detection exists. The actual recall killers are segmentation and
  area gating, not a candidate-level convexity test.

### Yellow cylinder (recall 0.54)

- H~38–40 and S~76–77 clear both overlay HSV sets and finals yellow S>=45.
- The curved/specular top can yield a small or disconnected high-V region. The finals hue vote always selects the
  brightest 35% of the surviving colour pixels; on a compact or split cylinder this subset can be dominated by
  glare/edge pixels rather than the majority yellow surface. A hue outside yellow or close to an adjacent red
  candidate is then abstained.
- Opening can disconnect a highlight-broken ring; each fragment separately faces the 120 px floor and, in finals,
  the distance-scaled area floor. A full cylinder projects to 78.5% of a cube, so it has more area margin than the
  prism but much less than the square model assumes.

### Occlusion by arm, hand, or another object

- Finals applies the FK mask before grouping. An overlapping object becomes one or more partial components; there
  is currently no record that a small component touched the mask. Each remnant must independently clear 120 px and
  the finals area floor, so a real object can vanish with no tentative result.
- rc1 has no FK mask, but the arm/hand can merge with a same-band object or physically hide enough coloured pixels
  to fail the same 120 px floor.
- Same-colour neighbours are split only if the merged component is >1.5 expected objects and every proposed part
  is >=0.3 expected. Failed splitting leaves a merged component. Finals can reject it above the 3.0x ceiling;
  successful splitting can still create occluded parts below the later 0.4x floor.

### Tape and bin rim controls

The observed zero tape false positives are explained by several independent controls: thin/low-S tape is damaged
by double opening, object HSV S minima still apply, components need 120 px, `_keep_mask` removes projected bin
hulls, `_reject` removes bin-centred blobs, and finals adds workspace/bin-edge and maximum-area gates. The recall
changes should retain the hue band requirement and these spatial/maximum-area controls rather than relying on a
fixed S=110 purple floor.
