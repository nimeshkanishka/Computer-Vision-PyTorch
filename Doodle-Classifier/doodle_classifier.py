import pygame
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from network import ClassifierNetwork

########## CONFIG ##########
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 750
CANVAS_SIZE = 650
BG_COLOR = (50, 50, 75)
CANVAS_COLOR = (0, 0, 0)
DRAW_COLOR = (255, 255, 255)
TEXT_COLOR_1 = (255, 255, 255)
TEXT_COLOR_2 = (150, 150, 150)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Label to class mapping
CLASSES = {
    0: "Airplane",
    1: "Angel",
    2: "Ant",
    3: "Apple",
    4: "Banana",
    5: "Bat",
    6: "Bird",
    7: "Book",
    8: "Bowtie",
    9: "Broccoli"
}
############################

# Load saved model checkpoint
model = ClassifierNetwork().to(DEVICE)
model.load_state_dict(torch.load("Doodle-Classifier/models/model_checkpoint.pth"))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("DrawIt! - Doodle Classifier")
clock = pygame.time.Clock()
font_large = pygame.font.SysFont("Arial", 50)
font_small = pygame.font.SysFont("Arial", 25)

canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill(CANVAS_COLOR)


def get_prediction(surface):
    # Convert pygame surface -> PIL -> Tensor
    data = pygame.surfarray.array3d(surface) # (W, H, C)
    data = np.transpose(data, (1, 0, 2)) # -> (H, W, C)

    img = Image.fromarray(data)

    img = transform(img).unsqueeze(dim=0).to(DEVICE) # -> (B, C, H, W)

    # Get model predictions
    with torch.no_grad():
        logits = model(img) # (1, 10)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy() # (10,)
    return probs


# Main loop
canvas_pos = (WINDOW_HEIGHT - CANVAS_SIZE) // 2
is_drawing, is_erasing = False, False
running = True
while running:
    # Background color
    screen.fill(BG_COLOR)

    # Draw canvas
    screen.blit(canvas, (canvas_pos, canvas_pos))

    # Handle events
    for event in pygame.event.get():
        # Quit
        if event.type == pygame.QUIT:
            running = False
        # Start drawing if LMB down
        # Start erasing if RMB down
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_drawing = True
            elif event.button == 3:
                is_erasing = True
        # Stop drawing if LMB up
        # Stop drawing if RMB up
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                is_drawing = False
            elif event.button == 3:
                is_erasing = False
        # Clear canvas if C key is pressed
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                canvas.fill(CANVAS_COLOR)

    # Drawing/Erasing
    if is_drawing or is_erasing:
        # Get mouse position on pygame screen
        x, y = pygame.mouse.get_pos()
        # Check if mouse is on the canvas
        if (canvas_pos <= x < canvas_pos + CANVAS_SIZE) and \
            (canvas_pos <= y < canvas_pos + CANVAS_SIZE):
            # Get mouse position on canvas
            pos = (x - 20, y - 20)
            color = DRAW_COLOR if is_drawing else CANVAS_COLOR
            pygame.draw.circle(canvas, color, pos, radius=20)

    # Get model predictions
    probs = get_prediction(canvas)
    top_indices = probs.argsort()[::-1]

    # Display predictions
    for i, idx in enumerate(top_indices[:10]):
        color = TEXT_COLOR_1 if i == 0 else TEXT_COLOR_2
        class_label = font_large.render(f"{CLASSES[idx]}", True, color)
        screen.blit(
            class_label,
            (canvas_pos + CANVAS_SIZE + 50, canvas_pos + 45 + i * 55)
        )
        confidence_label = font_large.render(f"{probs[idx] * 100:.2f}%", True, color)
        screen.blit(
            confidence_label,
            (canvas_pos + CANVAS_SIZE + 300, canvas_pos + 45 + i * 55)
        )

    # Display controls
    controls = font_small.render("LMB: Draw | RMB: Erase | C: Clear", True, TEXT_COLOR_1)
    screen.blit(
        controls,
        (canvas_pos + 160, canvas_pos + CANVAS_SIZE + 10)
    )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()