"""ForenSight: explainable multimodal deepfake and synthetic media detection."""

__version__ = "0.1.0"

# Label convention used everywhere in the project:
#   0 = real / bona fide, 1 = fake / spoofed.
# Model outputs are "fake scores": higher means more likely fake.
REAL = 0
FAKE = 1
