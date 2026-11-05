# Annotation Guide (CVAT)

Classes: `player`, `ball`.

## General rules
- Boxes are axis-aligned and as tight as possible around the visible object.
- Annotate only what is visible. Do not guess hidden body parts.
- Every box needs exactly one class.

## Players
- Include head to feet (and the visible body) in the box.
- Include goalkeepers as `player`.
- Do not annotate referees, coaches or crowd as `player`.
- **Partially visible:** annotate if at least roughly 30-40% of the body is
  visible and you are confident it is a player. Box only the visible part.
- **Overlapping players:** draw a separate box per player, even if the boxes
  overlap. Do not merge two players into one box.

## Ball
- Draw a small tight box around the ball itself, not its shadow or motion trail.
- **Blurred or tiny ball:** annotate only if you can identify it with
  confidence, using context from the player's feet and movement. Otherwise skip
  it.
- Only one ball is normally in play; ignore spare balls on the touchline.

## Unclear objects
If you cannot identify an object, do not annotate it. When many ambiguous cases
exist in an image, prefer excluding the image over adding uncertain labels.

## Quality review
1. A second pass checks 10% of images for missed objects and loose boxes.
2. Run the validation script (`src/data_quality.py`) before exporting.
3. Export from CVAT as COCO 1.0, then import with `feedback_manager.py`.
