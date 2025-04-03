from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
import os
from django.conf import settings
import uuid

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import base64
from io import BytesIO
import json
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_protect
from django.core.files.storage import FileSystemStorage



from torchvision.models import efficientnet_b0


class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, frame_count=80):
        super(DeepfakeDetectionModel, self).__init__()
        efficientnet = efficientnet_b0(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze feature extractor

        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, frames, channels, height, width = x.shape
        x = x.view(batch_size * frames, channels, height, width)
        features = self.feature_extractor(x)
        features = features.view(batch_size, frames, -1)

        lstm_out, _ = self.lstm(features)
        out = lstm_out[:, -1, :]
        return self.classifier(out)


# For training use a custom loss that emphasizes hard-to-detect fakes
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        ce_loss = self.criterion(outputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


# Video prediction function
def predict_video(video_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    frames, frame_paths, num_frames, original_frames, fps = extract_frames(video_path)
    print(f"📸 Total Frames Processed: {num_frames}")
    fake_scores = []

    # Process each frame individually to avoid LSTM issues
    with torch.no_grad():
        for i in range(num_frames):
            frame = frames[i].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, 224, 224] - Add batch and time dimensions
            output = model(frame)  # Predict
            prob = torch.softmax(output, dim=1)[:, 1].cpu().item()  # Get fake probability
            fake_scores.append(prob)

    # Separate fake and real frame scores
    fake_frame_scores = [score for score in fake_scores if score > 0.5]
    real_frame_scores = [score for score in fake_scores if score <= 0.5]

    # Count frames above and below threshold
    fake_frames = len(fake_frame_scores)
    real_frames = len(real_frame_scores)
    fake_percentage = (fake_frames / num_frames) * 100 if num_frames > 0 else 0
    real_percentage = (real_frames / num_frames) * 100 if num_frames > 0 else 0

    # Calculate averages
    avg_all_frames = np.mean(fake_scores) if fake_scores else 0
    avg_fake_frames = np.mean(fake_frame_scores) if fake_frame_scores else 0
    avg_real_frames = np.mean(real_frame_scores) if real_frame_scores else 0

    # Generate plot as base64 image
    frame_numbers = np.arange(1, num_frames + 1)
    fig = plt.figure(figsize=(12, 10))

    # Plot frame scores
    plt.subplot(2, 1, 1)
    bars = plt.bar(frame_numbers, fake_scores, alpha=0.7)
    # Color bars based on threshold
    for i, bar in enumerate(bars):
        bar.set_color('red' if fake_scores[i] > 0.5 else 'green')

    plt.xlabel("Frame Number")
    plt.ylabel("Fake Probability Score")
    plt.title(f"Fake Probability per Frame")
    plt.ylim(0, 1)  # Score between 0 and 1
    plt.xticks(frame_numbers[::max(1, num_frames // 10)], rotation=45)  # Adjust x-ticks dynamically
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Add reference lines for all averages
    plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.axhline(y=avg_all_frames, color='blue', linestyle='-', label=f'Avg All Frames: {avg_all_frames:.4f}')
    if fake_frames > 0:
        plt.axhline(y=avg_fake_frames, color='red', linestyle='-.', label=f'Avg Fake Frames: {avg_fake_frames:.4f}')
    if real_frames > 0:
        plt.axhline(y=avg_real_frames, color='green', linestyle='-.', label=f'Avg Real Frames: {avg_real_frames:.4f}')
    plt.legend()

    # Add frame count pie chart
    plt.subplot(2, 1, 2)
    plt.pie([fake_frames, real_frames],
            labels=[f'Fake Frames (avg: {avg_fake_frames:.4f})',
                    f'Real Frames (avg: {avg_real_frames:.4f})'],
            colors=['red', 'green'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.1, 0))
    plt.axis('equal')

    # Add comparison result to chart title
    comparison_result = "FAKE" if avg_fake_frames > avg_real_frames else "REAL"
    plt.title(
        f"Frame Classification: {comparison_result} (Avg Fake: {avg_fake_frames:.4f} vs Avg Real: {avg_real_frames:.4f})")

    plt.tight_layout()

    # Save plot to memory
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    # Calculate the difference between averages
    avg_difference = abs(avg_fake_frames - avg_real_frames)

    # Calculate absolute frame difference
    frame_difference = abs(real_frames - fake_frames)

    # Calculate confidence level
    if avg_difference > 0.5:
        confidence = "Very High"
    elif avg_difference > 0.3:
        confidence = "Medium"
    elif avg_difference > 0.2:
        confidence = "Low"
    elif avg_difference > 0.1:
        confidence = "Very Low"
    else:
        confidence = "Not Fake"

    # DECISION LOGIC:
    # 1. First priority: Compare avg_fake_frames with avg_real_frames
    # 2. Second priority: Consider frame count difference if it's significant (>50)

    # First check based on average frame scores (priority 1)
    avg_based_prediction = "Fake" if (avg_fake_frames - avg_real_frames) >= 0.15 else "Real"

    # Final decision (priority to average-based prediction)
    final_prediction = avg_based_prediction

    # Return detailed results
    result = {
        "prediction": final_prediction,
        "confidence": confidence,
        "fake_frames": fake_frames,
        "real_frames": real_frames,
        "fake_percentage": fake_percentage,
        "real_percentage": real_percentage,
        "frame_difference": frame_difference,
        "avg_fake_frames": avg_fake_frames,
        "avg_real_frames": avg_real_frames,
        "avg_difference": avg_difference,
        "plot_data": plot_data,
        "frame_paths": frame_paths[:min(20, len(frame_paths))],  # Limit to 20 frames
        "frame_scores": fake_scores[:min(20, len(fake_scores))],  # Scores for the frames
        "all_frames": num_frames,
        "fps": fps  # Return video fps
    }

    return result


# Frame extraction function
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms


def extract_frames(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_paths = []  # To store paths/identifiers for frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Match EfficientNet input size
        transforms.ToTensor()
    ])

    # If video has more frames, sample evenly
    frame_indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)

    actual_frames_processed = 0
    original_frames = []  # To store original frames for possible display

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if idx in frame_indices:
            # Store original frame for thumbnail display
            original_frames.append(frame.copy())

            # Process frame for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame_tensor = transform(frame_rgb)  # Apply transforms
            frames.append(frame_tensor)

            # Store a unique identifier for this frame
            frame_paths.append(f"frame_{idx}")
            actual_frames_processed += 1

    cap.release()

    # Ensure we return exactly max_frames (either crop or pad)
    while len(frames) < max_frames:
        frames.append(torch.zeros(3, 224, 224))  # Pad with black frames
        frame_paths.append(f"padding_{len(frame_paths)}")  # Add placeholder path

    # Return the frames tensor, frame paths list, actual frames processed,
    # original frames for thumbnails, and the video fps
    return torch.stack(frames), frame_paths, actual_frames_processed, original_frames, fps


# Video prediction function
# def predict_video(video_path, model):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#
#     frames, frame_paths, num_frames = extract_frames(video_path)
#     print(f"📸 Total Frames Processed: {num_frames}")
#     fake_scores = []
#
#     # Process frames in batch format (add a time dimension)
#     with torch.no_grad():
#         for i in range(num_frames):
#             frame = frames[i].unsqueeze(0).to(device)
#             if i == 0:
#                 # First frame, use it to determine if we need to process differently
#                 output = model(frame)
#                 if output.shape[1] == 2:  # If model expects single frame
#                     single_frame_mode = True
#                 else:
#                     single_frame_mode = False
#                     # Add time dimension for LSTM
#                     frame = frame.unsqueeze(0)  # [1, 1, 3, 224, 224]
#             else:
#                 if single_frame_mode:
#                     frame = frame  # Keep as [1, 3, 224, 224]
#                 else:
#                     frame = frame.unsqueeze(0)  # [1, 1, 3, 224, 224]
#
#             output = model(frame)  # Predict
#             prob = torch.softmax(output, dim=1)[:, 1].cpu().item()  # Get fake probability
#             fake_scores.append(prob)
#
#     # Separate fake and real frame scores
#     fake_frame_scores = [score for score in fake_scores if score > 0.5]
#     real_frame_scores = [score for score in fake_scores if score <= 0.5]
#
#     # Count frames above and below threshold
#     fake_frames = len(fake_frame_scores)
#     real_frames = len(real_frame_scores)
#     fake_percentage = (fake_frames / num_frames) * 100
#     real_percentage = (real_frames / num_frames) * 100
#
#     # Calculate averages
#     avg_all_frames = np.mean(fake_scores)
#     avg_fake_frames = np.mean(fake_frame_scores) if fake_frame_scores else 0
#     avg_real_frames = np.mean(real_frame_scores) if real_frame_scores else 0
#
#     # Generate plot as base64 image
#     frame_numbers = np.arange(1, num_frames + 1)
#     fig = plt.figure(figsize=(12, 10))
#
#     # Plot frame scores
#     plt.subplot(2, 1, 1)
#     bars = plt.bar(frame_numbers, fake_scores, alpha=0.7)
#     # Color bars based on threshold
#     for i, bar in enumerate(bars):
#         bar.set_color('red' if fake_scores[i] > 0.5 else 'green')
#
#     plt.xlabel("Frame Number")
#     plt.ylabel("Fake Probability Score")
#     plt.title(f"Fake Probability per Frame")
#     plt.ylim(0, 1)  # Score between 0 and 1
#     plt.xticks(frame_numbers[::max(1, num_frames // 10)], rotation=45)  # Adjust x-ticks dynamically
#     plt.grid(axis="y", linestyle="--", alpha=0.6)
#
#     # Add reference lines for all averages
#     plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')
#     plt.axhline(y=avg_all_frames, color='blue', linestyle='-', label=f'Avg All Frames: {avg_all_frames:.4f}')
#     if fake_frames > 0:
#         plt.axhline(y=avg_fake_frames, color='red', linestyle='-.', label=f'Avg Fake Frames: {avg_fake_frames:.4f}')
#     if real_frames > 0:
#         plt.axhline(y=avg_real_frames, color='green', linestyle='-.', label=f'Avg Real Frames: {avg_real_frames:.4f}')
#     plt.legend()
#
#     # Add frame count pie chart
#     plt.subplot(2, 1, 2)
#     plt.pie([fake_frames, real_frames],
#             labels=[f'Fake Frames (avg: {avg_fake_frames:.4f})',
#                     f'Real Frames (avg: {avg_real_frames:.4f})'],
#             colors=['red', 'green'],
#             autopct='%1.1f%%',
#             startangle=90,
#             explode=(0.1, 0))
#     plt.axis('equal')
#
#     # Add comparison result to chart title
#     comparison_result = "FAKE" if avg_fake_frames > avg_real_frames else "REAL"
#     plt.title(
#         f"Frame Classification: {comparison_result} (Avg Fake: {avg_fake_frames:.4f} vs Avg Real: {avg_real_frames:.4f})")
#
#     plt.tight_layout()
#
#     # Save plot to memory
#     buffer = BytesIO()
#     fig.savefig(buffer, format='png')
#     buffer.seek(0)
#     plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     plt.close(fig)
#
#     # Calculate the difference between averages
#     avg_difference = abs(avg_fake_frames - avg_real_frames)
#
#     # Calculate absolute frame difference
#     frame_difference = abs(real_frames - fake_frames)
#
#     # Calculate confidence level
#     if avg_difference > 0.5:
#         confidence = "Very High"
#     elif avg_difference > 0.3:
#         confidence = "Medium"
#     elif avg_difference > 0.2:
#         confidence = "Low"
#     elif avg_difference > 0.1:
#         confidence = "Very Low"
#     else:
#         confidence = "Not Fake"
#
#     # DECISION LOGIC:
#     # 1. First priority: Compare avg_fake_frames with avg_real_frames
#     # 2. Second priority: Consider frame count difference if it's significant (>50)
#
#     # First check based on average frame scores (priority 1)
#     avg_based_prediction = "Fake" if (avg_fake_frames - avg_real_frames) >= 0.15 else "Real"
#
#     # Then consider frame difference as secondary factor
#     if frame_difference >= 50 and real_frames > fake_frames:
#         frame_based_suggestion = "Real"
#     elif frame_difference >= 50 and fake_frames > real_frames:
#         frame_based_suggestion = "Fake"
#     else:
#         frame_based_suggestion = None
#
#     # Final decision (priority to average-based prediction)
#     final_prediction = avg_based_prediction
#
#     # Return detailed results
#     result = {
#         "prediction": final_prediction,
#         "confidence": confidence,
#         "fake_frames": fake_frames,
#         "real_frames": real_frames,
#         "fake_percentage": fake_percentage,
#         "real_percentage": real_percentage,
#         "frame_difference": frame_difference,
#         "avg_fake_frames": avg_fake_frames,
#         "avg_real_frames": avg_real_frames,
#         "avg_difference": avg_difference,
#         "plot_data": plot_data,
#         "frame_paths": frame_paths[:min(20, len(frame_paths))],  # Limit to 20 frames
#         "frame_scores": fake_scores[:min(20, len(fake_scores))],  # Scores for the frames
#         "all_frames": num_frames
#     }
#
#     return result


# Initialize model (load only once when the server starts)
def load_model():
    model_path = "D:\pycharm\DeepFakeCDI\deepfakedetection\myapp\Deep_Fake_model2.pth"
    model = DeepfakeDetectionModel()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model






def landing(request):
    return render(request,'landing.html')


def dashboard(request):
    return render(request,'dashboard.html')






# Global model instance
MODEL = None


# API endpoint to process video
# API endpoint to process video
@csrf_protect
def process_video(request):
    global MODEL

    # Load model if not already loaded
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception as e:
            return JsonResponse({
                "success": False,
                "error": f"Failed to load model: {str(e)}"
            })

    if request.method == "POST" and request.FILES.get("video"):
        try:
            video_file = request.FILES["video"]

            # Ensure directories exist
            os.makedirs(os.path.join(settings.MEDIA_ROOT, "videos"), exist_ok=True)

            # Save the uploaded video
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "videos"))
            filename = fs.save(video_file.name, video_file)
            video_path = os.path.join(settings.MEDIA_ROOT, "videos", filename)

            # Process the video with the model
            result = predict_video(video_path, MODEL)

            # Convert numpy values to Python native types for JSON serialization
            for key in result:
                if isinstance(result[key], np.float32) or isinstance(result[key], np.float64):
                    result[key] = float(result[key])
                elif isinstance(result[key], np.int32) or isinstance(result[key], np.int64):
                    result[key] = int(result[key])

            return JsonResponse({
                "success": True,
                "result": result
            })

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Error processing video: {str(e)}\n{error_traceback}")
            return JsonResponse({
                "success": False,
                "error": f"Error processing video: {str(e)}",
                "traceback": error_traceback
            })

    return JsonResponse({"success": False, "error": "Invalid request method or no video uploaded"})



from django.views.decorators.csrf import csrf_exempt

@csrf_protect
def upload_video(request):
    if request.method == 'GET':
        return render(request, 'dashboard.html')

    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)

        # Simulated processing
        detection_result = process_video(video_path)

        return JsonResponse({'status': 'success', 'result': detection_result, 'video_path': video_path})

    return JsonResponse({'status': 'failed', 'error': 'No video uploaded'}, status=400)
