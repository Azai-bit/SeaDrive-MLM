DINOv2 prototype library for ship-type classification.

Directory layout:
- Lifeboat/*.png
- USV/*.png
- Fishing/*.png

Current defaults are cropped from an existing simulated front-camera trigger image in tmp/.
They are only bootstrap references for the current Gazebo scene and can be replaced with your own views.

Recommendations:
- Keep 2 to 10 reference crops per class.
- Use front-camera crops with the boat centered and minimal water/background.
- Mix near, medium, and far views if you want more robust matching.

Runtime notes:
- The decision node uses DINOv2 only when llm_backend=dinov2.
- If torch/transformers or valid prototypes are unavailable, it falls back to a color/size heuristic and marks source=dinov2_fallback.
