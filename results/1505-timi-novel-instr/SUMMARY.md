# Dataset Summary: Novel Robot Evaluation Instructions

This dataset contains **60 diverse, realistic natural-language instructions** designed to test the **generality** of a humanoid robot tabletop sorting system. 

The task domain involves sorting 3–5 small colored objects (**purple**, **yellow**, and **orange** **cubes**, **cylinders**, and **triangular prisms**) into three bins (**left**, **center**, and **right**) or specific locations on the table.

---

## Action Primitive Grammar

All execution plans in the `expected` field are defined strictly using the following four primitives:

- `pick_place(obj, bin)`: Grasps the specified object and places it into the target bin (`left_bin`, `center_bin`, or `right_bin`).
- `place_at(obj, table location)`: Grasps the object and sets it down at an indicated location on the tabletop (e.g. `in_front_of_center_bin`, `table_center`).
- `skip(obj)`: Explicitly bypasses or leaves the designated object untouched.
- `done`: Signals the successful conclusion of the task plan.

---

## Category Breakdown & Counts

| Category | Count | Refuse (`should_refuse: true`) | Description & Test Focus |
| :--- | :---: | :---: | :--- |
| **`sequencing`** | 6 | 0 | Multi-step temporal execution ("first ... then ...", "before X do Y", staged ordering). |
| **`quantities`** | 5 | 0 | Cardinality and numerical constraints ("only two", "exactly one", "all three", "at most one"). |
| **`negation`** | 5 | 0 | Inversion and exclusion ("everything except yellow", "non-cube", "do not touch orange"). |
| **`relations`** | 6 | 0 | Spatial and relative references ("closest to you", "next to purple cube", "between X and Y"). |
| **`placing_on_table`** | 5 | 0 | Table manipulation rather than bin disposal ("in front of middle bin", "table center"). |
| **`swaps`** | 5 | 0 | Transposition of object states across bins or table positions. |
| **`conditional`** | 6 | 0 | Branching logic and state verification ("if cube present then X else Y"). |
| **`typos`** | 5 | 0 | Robustness against judge orthographic errors, misspellings, and omitted vowels. |
| **`mixed_languages`** | 6 | 0 | Multilingual code-switching incorporating Finnish and Czech vocabulary. |
| **`vague`** | 5 | 0 | High-level goal specifications requiring autonomous resolution ("tidy up", "clear workspace"). |
| **`impossible_unsafe`** | 6 | 6 | Unsafe, hazardous, or physically impossible requests where the robot must refuse. |
| **Total** | **60** | **6** | |

---

## Multilingual Vocabulary Details

The `mixed_languages` category includes natural bilingual prompts with authentic words:
- **Finnish**:
  - `keltainen` (yellow)
  - `oranssi` (orange)
  - `violetti` (purple)
  - `kuutio` (cube)
  - `keskimmäinen laatikko` (middle box / center bin)
  - `laita` (put)
  - `vie` (take)
  - `älä koske` (do not touch)
- **Czech**:
  - `žlutou kostku` (yellow cube)
  - `oranžový válec` (orange cylinder)
  - `fialový válec` (purple cylinder)
  - `fialový hranol` (purple prism)
  - `do levého koše` (into the left bin)
  - `do pravého koše` (into the right bin)
  - `na stůl vpravo` (on the table to the right)
  - `dej` (put)
  - `polož` (place)

---

## File Format

Each record in [NOVEL_INSTRUCTIONS.jsonl](file:///private/tmp/claude-501/-Users-timi-projects-hackjunction/228e0431-2379-43e8-adbd-ebe5e051700a/scratchpad/agy_planner/NOVEL_INSTRUCTIONS.jsonl) is serialized as a single-line JSON object:

```json
{
  "text": "First put the purple cube into the left bin, and then put the yellow cylinder into the center bin.",
  "category": "sequencing",
  "expected": "pick_place(purple_cube, left_bin); pick_place(yellow_cylinder, center_bin); done",
  "should_refuse": false
}
```
