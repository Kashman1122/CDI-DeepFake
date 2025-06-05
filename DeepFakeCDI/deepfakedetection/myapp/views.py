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
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
import os
from django.conf import settings
import uuid
import traceback
from django.urls import reverse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .metadata import VideoMetadataExtractor
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
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .models import UserProfile, VideoAnalysis, VectorDB
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from torchvision.models import efficientnet_b0
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from django.views.decorators.csrf import csrf_exempt

MODEL = None


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
# API endpoint to process video
# API endpoint to process video
# @csrf_protect
# @csrf_exempt
# def process_video(request):
#     global MODEL
#
#     # Load model if not already loaded
#     if MODEL is None:
#         try:
#             MODEL = load_model()
#         except Exception as e:
#             return JsonResponse({
#                 "success": False,
#                 "error": f"Failed to load model: {str(e)}"
#             })
#
#     if request.method == "POST" and request.FILES.get("video"):
#         try:
#             video_file = request.FILES["video"]
#             username = request.POST.get("username")
#
#             # Get the user (should be logged in at this point)
#             try:
#                 user = User.objects.get(username=username)
#             except User.DoesNotExist:
#                 return JsonResponse({
#                     "success": False,
#                     "error": "User not found. Please log in again."
#                 })
#
#             # Ensure directories exist
#             os.makedirs(os.path.join(settings.MEDIA_ROOT, "videos"), exist_ok=True)
#
#             # Save the uploaded video
#             fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "videos"))
#             filename = fs.save(video_file.name, video_file)
#             video_path = os.path.join(settings.MEDIA_ROOT, "videos", filename)
#
#             # Process the video with the model
#             result = predict_video(video_path, MODEL)
#
#             # Convert numpy values to Python native types for JSON serialization
#             for key in result:
#                 if isinstance(result[key], np.float32) or isinstance(result[key], np.float64):
#                     result[key] = float(result[key])
#                 elif isinstance(result[key], np.int32) or isinstance(result[key], np.int64):
#                     result[key] = int(result[key])
#
#             # Save analysis results to database
#             video_analysis = VideoAnalysis.objects.create(
#                 user=user,
#                 video_path=video_path,
#                 prediction=result.get('prediction', 'Unknown'),
#                 confidence=result.get('confidence', '0'),
#                 fake_percentage=result.get('fake_percentage', 0.0),
#                 real_percentage=result.get('real_percentage', 0.0),
#                 avg_fake_frames=result.get('avg_fake_frames', 0.0),
#                 avg_real_frames=result.get('avg_real_frames', 0.0),
#                 frame_count=result.get('frame_count', 0)
#             )
#
#             # Save frame embeddings if available
#             if 'frame_embeddings' in result:
#                 VectorDB.objects.create(
#                     user=user,
#                     video_analysis=video_analysis,
#                     frame_embeddings=result['frame_embeddings']
#                 )
#
#             return JsonResponse({
#                 "success": True,
#                 "result": result
#             })
#
#         except Exception as e:
#             error_traceback = traceback.format_exc()
#             print(f"Error processing video: {str(e)}\n{error_traceback}")
#             return JsonResponse({
#                 "success": False,
#                 "error": f"Error processing video: {str(e)}",
#                 "traceback": error_traceback
#             })


# @csrf_protect
# @csrf_exempt
# def process_video(request):
#     global MODEL
#
#     # Load model if not already loaded
#     if MODEL is None:
#         try:
#             MODEL = load_model()
#         except Exception as e:
#             return JsonResponse({
#                 "success": False,
#                 "error": f"Failed to load model: {str(e)}"
#             })
#
#     if request.method == "POST" and request.FILES.get("video"):
#         try:
#             video_file = request.FILES["video"]
#
#             # Get the user (use authenticated user)
#             user = request.user
#             if not user.is_authenticated:
#                 # For development/testing, you could use a default user
#                 # Comment this out in production and use the line above that returns an error
#                 user = User.objects.get_or_create(username="default_user")[0]
#                 # Or return an error:
#                 # return JsonResponse({
#                 #    "success": False,
#                 #    "error": "User not authenticated. Please log in."
#                 # })
#
#             # Ensure directories exist
#             os.makedirs(os.path.join(settings.MEDIA_ROOT, "videos"), exist_ok=True)
#             os.makedirs(os.path.join(settings.MEDIA_ROOT, "frames"), exist_ok=True)
#
#             # Save the uploaded video
#             fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "videos"))
#             filename = fs.save(video_file.name, video_file)
#             video_path = os.path.join(settings.MEDIA_ROOT, "videos", filename)
#
#             # Process the video with the model
#             result = predict_video(video_path, MODEL)
#
#             # Add video URL to result
#             result["video_url"] = f"/media/videos/{filename}"
#
#             # Convert numpy values to Python native types for JSON serialization
#             for key in result:
#                 if isinstance(result[key], np.float32) or isinstance(result[key], np.float64):
#                     result[key] = float(result[key])
#                 elif isinstance(result[key], np.int32) or isinstance(result[key], np.int64):
#                     result[key] = int(result[key])
#
#             # Save analysis results to database
#             video_analysis = VideoAnalysis.objects.create(
#                 user=user,
#                 video_path=video_path,
#                 prediction=result.get('prediction', 'Unknown'),
#                 confidence=result.get('confidence', '0'),
#                 fake_percentage=result.get('fake_percentage', 0.0),
#                 real_percentage=result.get('real_percentage', 0.0),
#                 avg_fake_frames=result.get('avg_fake_frames', 0.0),
#                 avg_real_frames=result.get('avg_real_frames', 0.0),
#                 frame_count=result.get('all_frames', 0)  # Make sure field names match
#             )
#
#             return JsonResponse({
#                 "success": True,
#                 "result": result
#             })
#
#         except Exception as e:
#             error_traceback = traceback.format_exc()
#             print(f"Error processing video: {str(e)}\n{error_traceback}")
#             return JsonResponse({
#                 "success": False,
#                 "error": f"Error processing video: {str(e)}",
#                 "traceback": error_traceback
#             })
#
#     return JsonResponse({"success": False, "error": "Invalid request method or no video uploaded"})
#Upload Video means Whenever Video uploads starts from here
# @csrf_protect
# def upload_video(request):
#     if request.method == 'GET':
#         return render(request, 'dashboard.html')
#
#     if request.method == 'POST' and request.FILES.get('video'):
#         video_file = request.FILES['video']
#         fs = FileSystemStorage()
#         filename = fs.save(video_file.name, video_file)
#         video_path = fs.path(filename)
#
#         # Simulated processing
#         detection_result = process_video(video_path)
#
#         return JsonResponse({'status': 'success', 'result': detection_result, 'video_path': video_path})
#
#     return JsonResponse({'status': 'failed', 'error': 'No video uploaded'}, status=400)##
#contain Upload video view function

from pymediainfo import MediaInfo


#Video Upload here in the process_video function
# @csrf_protect
# @csrf_exempt
# def process_video(request):
#     global MODEL
#
#     # Load model if not already loaded
#     if MODEL is None:
#         try:
#             MODEL = load_model()
#         except Exception as e:
#             return JsonResponse({
#                 "success": False,
#                 "error": f"Failed to load model: {str(e)}"
#             })
#
#     if request.method == "POST" and request.FILES.get("video"):
#         try:
#             video_file = request.FILES["video"]
#
#             # Get the user (use authenticated user)
#             user = request.user
#             if not user.is_authenticated:
#                 # For development/testing, you could use a default user
#                 # Comment this out in production and use the line above that returns an error
#                 user = User.objects.get_or_create(username="default_user")[0]
#                 # Or return an error:
#                 # return JsonResponse({
#                 #    "success": False,
#                 #    "error": "User not authenticated. Please log in."
#                 # })
#
#             # Ensure directories exist
#             os.makedirs(os.path.join(settings.MEDIA_ROOT, "videos"), exist_ok=True)
#             os.makedirs(os.path.join(settings.MEDIA_ROOT, "frames"), exist_ok=True)
#
#             # Save the uploaded video
#             fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "videos"))
#             filename = fs.save(video_file.name, video_file)
#             video_path = os.path.join(settings.MEDIA_ROOT, "videos", filename)
#
#             # Process the video with the model
#             result = predict_video(video_path, MODEL)
#
#             # Add video URL to result
#             result["video_url"] = f"/media/videos/{filename}"
#
#             # Extract metadata using VideoMetadataExtractor
#             try:
#                 extractor = VideoMetadataExtractor()
#                 metadata = extractor.extract_from_local_file(video_path)
#
#                 # Organize metadata into categories for dashboard display
#                 metadata_categories = {
#                     "basic_details": {
#                         "filename": metadata.get("filename", "Unknown"),
#                         "file_size": metadata.get("file_size_human", "Unknown"),
#                         "format": metadata.get("ffmpeg", {}).get("format", {}).get("format_name", "Unknown"),
#                         "duration": metadata.get("ffmpeg", {}).get("format", {}).get("duration", "Unknown"),
#                     },
#                     "video_info": metadata.get("video_info", {}),
#                     "audio_info": metadata.get("audio_info", {}),
#                     "creation_history": metadata.get("creation_history", {}),
#                     "technical_metadata": {
#                         "codecs": metadata.get("ffmpeg", {}).get("format", {}).get("format_long_name", "Unknown"),
#                         "bit_rate": metadata.get("ffmpeg", {}).get("format", {}).get("bit_rate", "Unknown"),
#                         "start_time": metadata.get("ffmpeg", {}).get("format", {}).get("start_time", "Unknown"),
#                         "nb_streams": metadata.get("ffmpeg", {}).get("format", {}).get("nb_streams", "Unknown"),
#                     }
#                 }
#
#                 # Add metadata to result
#                 result["metadata"] = metadata_categories
#
#                 # Clean up temp files
#                 extractor.cleanup()
#
#             except Exception as metadata_error:
#                 # Log the error but continue processing
#                 print(f"Error extracting metadata: {str(metadata_error)}")
#                 result["metadata_error"] = str(metadata_error)
#
#             # Convert numpy values to Python native types for JSON serialization
#             for key in result:
#                 if isinstance(result[key], np.float32) or isinstance(result[key], np.float64):
#                     result[key] = float(result[key])
#                 elif isinstance(result[key], np.int32) or isinstance(result[key], np.int64):
#                     result[key] = int(result[key])
#
#             # Save analysis results to database
#             video_analysis = VideoAnalysis.objects.create(
#                 user=user,
#                 video_path=video_path,
#                 prediction=result.get('prediction', 'Unknown'),
#                 confidence=result.get('confidence', '0'),
#                 fake_percentage=result.get('fake_percentage', 0.0),
#                 real_percentage=result.get('real_percentage', 0.0),
#                 avg_fake_frames=result.get('avg_fake_frames', 0.0),
#                 avg_real_frames=result.get('avg_real_frames', 0.0),
#                 frame_count=result.get('all_frames', 0),
#                 metadata_json=json.dumps(result.get("metadata", {}), default=str)  # Store metadata in the database
#             )
#
#             # Check if user is paid and handle redirection accordingly
#             # You'll need to implement a way to check if a user is paid
#             # This example assumes a UserProfile model with a is_paid field
#             try:
#                 is_paid_user = hasattr(user, 'userprofile') and user.userprofile.is_paid
#             except:
#                 # If there's an error checking paid status, default to non-paid
#                 is_paid_user = False
#
#             if is_paid_user:
#                 result["redirect_url"] = reverse('dashboard')  # Paid users go to regular dashboard
#             else:
#                 result["redirect_url"] = reverse('temp_dashboard')  # Free users go to temp dashboard
#
#             # Make sure the complete result including the plot_data and metadata is returned
#             return JsonResponse({
#                 "success": True,
#                 "result": result,  # This includes plot_data, frame_paths, frame_scores, metadata, etc.
#                 "is_paid_user": is_paid_user  # Flag to indicate if user is paid
#             })
#
#         except Exception as e:
#             error_traceback = traceback.format_exc()
#             print(f"Error processing video: {str(e)}\n{error_traceback}")
#             return JsonResponse({
#                 "success": False,
#                 "error": f"Error processing video: {str(e)}",
#                 "traceback": error_traceback
#             })
#
#     return JsonResponse({"success": False, "error": "Invalid request method or no video uploaded"})

@csrf_protect
@csrf_exempt
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

            # Get the user (use authenticated user)
            user = request.user
            if not user.is_authenticated:
                # For development/testing, you could use a default user
                # Comment this out in production and use the line above that returns an error
                user = User.objects.get_or_create(username="default_user")[0]
                # Or return an error:
                # return JsonResponse({
                #    "success": False,
                #    "error": "User not authenticated. Please log in."
                # })

            # Ensure directories exist
            os.makedirs(os.path.join(settings.MEDIA_ROOT, "videos"), exist_ok=True)
            os.makedirs(os.path.join(settings.MEDIA_ROOT, "frames"), exist_ok=True)

            # Save the uploaded video
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "videos"))
            filename = fs.save(video_file.name, video_file)
            video_path = os.path.join(settings.MEDIA_ROOT, "videos", filename)

            # Process the video with the model
            result = predict_video(video_path, MODEL)

            # Add video URL to result
            result["video_url"] = f"/media/videos/{filename}"

            # Lists of known deepfake tools and video editors
            deepfake_tools = [
                "deepfacelab", "faceswap", "deepfakesweb", "deepbrain", "colossyan",
                "reface", "zao", "wombo", "face swap", "heygen", "ltx studio", "lightricks",
                "video gen", "omnihuman", "bytedance"
            ]

            video_editors = [
                "davinci resolve", "premiere pro", "final cut pro", "hitfilm", "lightworks",
                "shotcut", "openshot", "lumafusion", "capcut", "premiere rush", "inshot",
                "kinemaster", "youcut", "adobe express", "canva", "clipchamp", "kapwing",
                "flexclip", "veed", "invideo"
            ]

            # Initialize variables to track detected software
            detected_deepfake_tools = []
            detected_video_editors = []
            software_detected = False

            # Extract metadata using VideoMetadataExtractor
            try:
                extractor = VideoMetadataExtractor()
                metadata = extractor.extract_from_local_file(video_path)

                # Organize metadata into categories for dashboard display
                metadata_categories = {
                    "basic_details": {
                        "filename": metadata.get("filename", "Unknown"),
                        "file_size": metadata.get("file_size_human", "Unknown"),
                        "format": metadata.get("ffmpeg", {}).get("format", {}).get("format_name", "Unknown"),
                        "duration": metadata.get("ffmpeg", {}).get("format", {}).get("duration", "Unknown"),
                    },
                    "video_info": metadata.get("video_info", {}),
                    "audio_info": metadata.get("audio_info", {}),
                    "creation_history": metadata.get("creation_history", {}),
                    "technical_metadata": {
                        "codecs": metadata.get("ffmpeg", {}).get("format", {}).get("format_long_name", "Unknown"),
                        "bit_rate": metadata.get("ffmpeg", {}).get("format", {}).get("bit_rate", "Unknown"),
                        "start_time": metadata.get("ffmpeg", {}).get("format", {}).get("start_time", "Unknown"),
                        "nb_streams": metadata.get("ffmpeg", {}).get("format", {}).get("nb_streams", "Unknown"),
                    }
                }

                # Extract software info using pymediainfo
                try:
                    from pymediainfo import MediaInfo

                    # Function to extract software info
                    def get_video_software_info(file_path):
                        software_info = {}
                        media_info = MediaInfo.parse(file_path)
                        found = False
                        detected_software = []

                        for track in media_info.tracks:
                            if track.track_type == "General":
                                software_info["file_path"] = file_path
                                software_info["format"] = track.format if hasattr(track, 'format') else "Unknown"

                                # Check for common software info fields
                                if hasattr(track, 'encoded_by') and track.encoded_by:
                                    software_info["encoded_by"] = track.encoded_by
                                    detected_software.append(track.encoded_by)
                                    found = True

                                if hasattr(track, 'writing_application') and track.writing_application:
                                    software_info["writing_application"] = track.writing_application
                                    detected_software.append(track.writing_application)
                                    found = True

                                if hasattr(track, 'writing_library') and track.writing_library:
                                    software_info["writing_library"] = track.writing_library
                                    detected_software.append(track.writing_library)
                                    found = True

                                # Get all available metadata if no explicit software info found
                                if not found:
                                    additional_info = {}
                                    for key, value in track.to_data().items():
                                        if isinstance(value, str) and any(
                                                kw in key.lower() for kw in
                                                ["application", "encoder", "software", "library", "tool"]):
                                            additional_info[key] = value
                                            detected_software.append(value)

                                    if additional_info:
                                        software_info["additional_software_metadata"] = additional_info
                                        found = True

                                # Check for specific markers
                                track_data_str = str(track.to_data())
                                if "Lavf57.83.100" in track_data_str:
                                    software_info["ffmpeg_marker"] = "Lavf57.83.100"
                                    software_info[
                                        "risk_assessment"] = "High risk - Contains specific ffmpeg signature associated with fake videos"
                                    found = True
                                    detected_software.append("ffmpeg")
                                elif any(ff_marker in track_data_str for ff_marker in ["Lavf", "ffmpeg"]):
                                    software_info["ffmpeg_marker"] = True
                                    software_info[
                                        "risk_assessment"] = "Elevated risk - ffmpeg tools often used in video manipulation"
                                    found = True
                                    detected_software.append("ffmpeg")

                                break

                        if not found:
                            software_info["status"] = "No software information found"

                        # Return both the software info and detected software names
                        return software_info, detected_software

                    # Get software info and add to metadata categories
                    software_info, detected_software = get_video_software_info(video_path)
                    metadata_categories["software_info"] = software_info

                    # Check for deepfake tools and video editors in detected software
                    metadata_str = json.dumps(metadata).lower()
                    software_str = " ".join(detected_software).lower()

                    # Also check filename for hints about software used
                    filename_lower = filename.lower()

                    # Combined string to search for software mentions
                    all_text = f"{metadata_str} {software_str} {filename_lower}"

                    # Check for deepfake tools
                    for tool in deepfake_tools:
                        if tool.lower() in all_text:
                            detected_deepfake_tools.append(tool)
                            software_detected = True

                    # Check for video editors
                    for editor in video_editors:
                        if editor.lower() in all_text:
                            detected_video_editors.append(editor)
                            software_detected = True

                    # Add detected software to metadata
                    if detected_deepfake_tools or detected_video_editors:
                        metadata_categories["detected_software"] = {
                            "deepfake_tools": detected_deepfake_tools,
                            "video_editors": detected_video_editors
                        }

                except ImportError:
                    metadata_categories["software_info"] = {"error": "pymediainfo not available"}
                except Exception as software_info_error:
                    metadata_categories["software_info"] = {
                        "error": f"Failed to extract software info: {str(software_info_error)}"}

                # Add metadata to result
                result["metadata"] = metadata_categories

                # Clean up temp files
                extractor.cleanup()

            except Exception as metadata_error:
                # Log the error but continue processing
                print(f"Error extracting metadata: {str(metadata_error)}")
                result["metadata_error"] = str(metadata_error)

            # Adjust fake percentage based on detected software
            original_fake_percentage = result.get('fake_percentage', 0.0)
            original_real_percentage = result.get('real_percentage', 100.0 - original_fake_percentage)

            # Initialize adjusted percentages
            adjusted_fake_percentage = original_fake_percentage

            # Apply adjustments based on detected software
            if detected_deepfake_tools:
                # Increase fake percentage by 40% if deepfake tools detected
                adjusted_fake_percentage = min(100.0, original_fake_percentage + 80.0)
                result["adjustment_reason"] = f"Deepfake tool(s) detected: {', '.join(detected_deepfake_tools)}"
            elif detected_video_editors:
                # Increase fake percentage by 20% if video editors detected
                adjusted_fake_percentage = min(100.0, original_fake_percentage + 70.0)
                result["adjustment_reason"] = f"Video editor(s) detected: {', '.join(detected_video_editors)}"

            # Update the result with adjusted percentages
            if software_detected:
                # Store original values for reference
                result["original_fake_percentage"] = original_fake_percentage
                result["original_real_percentage"] = original_real_percentage

                # Update with new values
                result["fake_percentage"] = adjusted_fake_percentage
                result["real_percentage"] = max(0.0, 100.0 - adjusted_fake_percentage)

                # Also update prediction if necessary based on the new percentages
                if adjusted_fake_percentage > 50.0 and result.get('prediction') == "Real":
                    result["prediction"] = "Fake"
                    result["prediction_changed"] = True

            # Convert numpy values to Python native types for JSON serialization
            for key in result:
                if isinstance(result[key], np.float32) or isinstance(result[key], np.float64):
                    result[key] = float(result[key])
                elif isinstance(result[key], np.int32) or isinstance(result[key], np.int64):
                    result[key] = int(result[key])

            # Ensure fake_scores are properly formatted for frontend
            if 'fake_scores' in result and isinstance(result['fake_scores'], list):
                # Convert numpy values to regular Python floats for JSON serialization
                result['fake_scores'] = [float(score) if isinstance(score, (np.float32, np.float64)) else score
                                         for score in result['fake_scores']]

                # Create structured data for line chart
                result['chart_data'] = {
                    'frame_numbers': list(range(1, len(result['fake_scores']) + 1)),
                    'fake_scores': result['fake_scores'],
                    'threshold': 0.5,  # Reference line for threshold
                    'avg_score': float(np.mean(result['fake_scores'])) if result['fake_scores'] else 0.0
                }

            # Save analysis results to database
            video_analysis = VideoAnalysis.objects.create(
                user=user,
                video_path=video_path,
                prediction=result.get('prediction', 'Unknown'),
                confidence=result.get('confidence', '0'),
                fake_percentage=result.get('fake_percentage', 0.0),
                real_percentage=result.get('real_percentage', 0.0),
                avg_fake_frames=result.get('avg_fake_frames', 0.0),
                avg_real_frames=result.get('avg_real_frames', 0.0),
                frame_count=result.get('all_frames', 0),
                metadata_json=json.dumps(result.get("metadata", {}), default=str),  # Store metadata in the database
                fake_scores_json=json.dumps(result.get('fake_scores', []))  # Store fake_scores separately
            )

            # Check if user is paid and handle redirection accordingly
            # You'll need to implement a way to check if a user is paid
            # This example assumes a UserProfile model with a is_paid field
            try:
                is_paid_user = hasattr(user, 'userprofile') and user.userprofile.is_paid
            except:
                # If there's an error checking paid status, default to non-paid
                is_paid_user = False

            if is_paid_user:
                result["redirect_url"] = reverse('dashboard')  # Paid users go to regular dashboard
            else:
                result["redirect_url"] = reverse('temp_dashboard')  # Free users go to temp dashboard

            # Make sure the complete result including the plot_data and metadata is returned
            return JsonResponse({
                "success": True,
                "result": result,
                # This includes plot_data, frame_paths, frame_scores, metadata, fake_scores, chart_data etc.
                "is_paid_user": is_paid_user  # Flag to indicate if user is paid
            })

        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error processing video: {str(e)}\n{error_traceback}")
            return JsonResponse({
                "success": False,
                "error": f"Error processing video: {str(e)}",
                "traceback": error_traceback
            })

    return JsonResponse({"success": False, "error": "Invalid request method or no video uploaded"})


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

    print("Total Fake Score: ",fake_scores)
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
    # Using the original threshold of 0.15 for the difference between avg_fake_frames and avg_real_frames
    avg_based_prediction = "Fake" if (avg_fake_frames - avg_real_frames) >= 0.15 else "Real"

    # Final decision (priority to average-based prediction)
    final_prediction = avg_based_prediction

    # Extract metadata using VideoMetadataExtractor
    metadata_categories = {}
    try:
        extractor = VideoMetadataExtractor()
        metadata = extractor.extract_from_local_file(video_path)

        # Organize metadata into categories for dashboard display
        metadata_categories = {
            "basic_details": {
                "filename": metadata.get("filename", "Unknown"),
                "file_size": metadata.get("file_size_human", "Unknown"),
                "format": metadata.get("ffmpeg", {}).get("format", {}).get("format_name", "Unknown"),
                "duration": metadata.get("ffmpeg", {}).get("format", {}).get("duration", "Unknown"),
            },
            "video_info": metadata.get("video_info", {}),
            "audio_info": metadata.get("audio_info", {}),
            "creation_history": metadata.get("creation_history", {}),
            "technical_metadata": {
                "codecs": metadata.get("ffmpeg", {}).get("format", {}).get("format_long_name", "Unknown"),
                "bit_rate": metadata.get("ffmpeg", {}).get("format", {}).get("bit_rate", "Unknown"),
                "start_time": metadata.get("ffmpeg", {}).get("format", {}).get("start_time", "Unknown"),
                "nb_streams": metadata.get("ffmpeg", {}).get("format", {}).get("nb_streams", "Unknown"),
            }
        }

        # Clean up temp files
        extractor.cleanup()

    except Exception as metadata_error:
        # Log the error but continue processing
        print(f"Error extracting metadata: {str(metadata_error)}")

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
        "fps": fps,  # Return video fps
        "metadata": metadata_categories, # Add metadata to the result
        "fake_scores": fake_scores
    }

    return result


# def temp_dashboard(request):
#     if not request.user.is_authenticated:
#         return redirect('login')
#
#     try:
#         latest_analysis = VideoAnalysis.objects.filter(user=request.user).latest('created_at')
#
#         # Check if user is paid
#         try:
#             is_paid_user = hasattr(request.user, 'userprofile') and request.user.userprofile.is_paid
#         except:
#             is_paid_user = False
#
#         # If user is paid, redirect to main dashboard
#         if is_paid_user:
#             return redirect('dashboard')
#
#         metadata_json = json.dumps(result.get("metadata", {}), default=lambda x: str(x) if isinstance(x, (datetime.datetime, datetime.date)) else None)
#
#         context = {
#             'analysis': {
#                 'user':latest_analysis.user,
#                 'prediction': latest_analysis.prediction,
#                 'confidence': latest_analysis.confidence,
#                 'fake_percentage': latest_analysis.fake_percentage,
#                 'real_percentage': latest_analysis.real_percentage,
#                 'frame_count': latest_analysis.frame_count,
#                 'avg_fake_frames': latest_analysis.avg_fake_frames,
#                 'avg_real_frames': latest_analysis.avg_real_frames,
#                 'video_url': '/media/' + latest_analysis.video_path.split('media/')[
#                     1] if 'media/' in latest_analysis.video_path else latest_analysis.video_path,
#                 'created_at': latest_analysis.created_at,
#                 'metadata': metadata
#             }
#         }
#
#         return render(request, 'temp-dashboard.html', context)
#
#     except VideoAnalysis.DoesNotExist:
#         # If no analysis exists, redirect to home page or appropriate page
#         messages.warning(request, "Please upload a video for analysis first.")
#         return redirect('dashboard')  # Change to an appropriate URL


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


# def register_view(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         email = request.POST.get('email')
#         phone = request.POST.get('phone')
#         password = request.POST.get('password')
#
#         # Check if user already exists
#         if User.objects.filter(username=username).exists():
#             return JsonResponse({'status': 'error', 'message': 'Username already exists'})
#
#         if User.objects.filter(email=email).exists():
#             return JsonResponse({'status': 'error', 'message': 'Email already exists'})
#
#         # Create user
#         user = User.objects.create_user(username=username, email=email, password=password)
#
#         # Create user profile
#         UserProfile.objects.create(user=user, phone=phone)
#
#         # Login the user
#         login(request, user)
#
#         return JsonResponse({'status': 'success', 'redirect': '/dashboard/'})
#
#     return render(request, 'register.html')

@csrf_protect
def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        password = request.POST.get('password')

        # Check if user already exists
        if User.objects.filter(username=username).exists():
            return JsonResponse({'status': 'error', 'message': 'Username already exists'})

        if User.objects.filter(email=email).exists():
            return JsonResponse({'status': 'error', 'message': 'Email already exists'})

        # Create user
        user = User.objects.create_user(username=username, email=email, password=password)

        # Create user profile
        UserProfile.objects.create(user=user, phone=phone)

        # Login the user
        login(request, user)

        return JsonResponse({'status': 'success'})

    return render(request, 'register.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return JsonResponse({'status': 'success', 'redirect': '/temp_dashboard/'})  # Changed redirect to temp_dashboard
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid credentials'})

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard(request):
    # Check if user is paid
    try:
        # Using hasattr to prevent errors if userprofile doesn't exist
        is_paid_user = hasattr(request.user, 'userprofile') and request.user.userprofile.is_paid
    except:
        is_paid_user = False

    # If not a paid user, redirect to temp_dashboard
    if not is_paid_user:
        messages.warning(request, "This dashboard is only available for paid users.")
        return redirect('temp_dashboard')

    # Get the current user's analysis results
    analysis_results = VideoAnalysis.objects.filter(user=request.user).order_by('-created_at')

    context = {
        'analysis_results': analysis_results,
        'latest_analysis': analysis_results.first() if analysis_results.exists() else None
    }

    return render(request, 'dashboard.html', context)


@csrf_exempt
@login_required
def process_video_endpoint(request):
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
            user_video_dir = os.path.join(settings.MEDIA_ROOT, "videos", request.user.username)
            os.makedirs(user_video_dir, exist_ok=True)

            # Save the uploaded video
            fs = FileSystemStorage(location=user_video_dir)
            filename = fs.save(video_file.name, video_file)
            video_path = os.path.join(user_video_dir, filename)

            # Process the video
            result = process_video_helper(video_path, request.user)

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


def process_video_helper(video_path, user):
    global MODEL

    # Load model if not already loaded
    if MODEL is None:
        MODEL = load_model()

    # Process video
    result = predict_video(video_path, MODEL)

    # Convert numpy values to Python native types for JSON serialization
    for key in result:
        if isinstance(result[key], (np.float32, np.float64)):
            result[key] = float(result[key])
        elif isinstance(result[key], (np.int32, np.int64)):
            result[key] = int(result[key])

    # Save analysis to database
    analysis = VideoAnalysis.objects.create(
        user=user,
        video_path=video_path,
        prediction=result['prediction'],
        confidence=result['confidence'],
        fake_percentage=result['fake_percentage'],
        real_percentage=result['real_percentage'],
        avg_fake_frames=result['avg_fake_frames'],
        avg_real_frames=result['avg_real_frames'],
        frame_count=result['all_frames']
    )

    # Create sample embeddings (in real application, you'd extract these from your model)
    sample_embeddings = {
        "frame0": [0.1, 0.2, 0.3],  # This is just a placeholder
        "frame1": [0.2, 0.3, 0.4]
    }

    # Save to VectorDB
    VectorDB.objects.create(
        user=user,
        video_analysis=analysis,
        frame_embeddings=sample_embeddings
    )

    return result


def pricing(request):
    return render(request,'pricing.html')


# def temp_dashboard(request):
#     if not request.user.is_authenticated:
#         return redirect('login')
#
#     try:
#         latest_analysis = VideoAnalysis.objects.filter(user=request.user).latest('created_at')
#
#         # Check if user is paid
#         try:
#             is_paid_user = hasattr(request.user, 'userprofile') and request.user.userprofile.is_paid
#         except:
#             is_paid_user = False
#
#         # If user is paid, redirect to main dashboard
#         if is_paid_user:
#             return redirect('dashboard')
#
#         # Parse the metadata from the stored JSON
#         try:
#             metadata = json.loads(latest_analysis.metadata_json)
#         except (json.JSONDecodeError, AttributeError):
#             metadata = {}
#
#         context = {
#             'analysis': {
#                 'prediction': latest_analysis.prediction,  # Make sure this value is coming through correctly
#                 'confidence': latest_analysis.confidence,
#                 'fake_percentage': latest_analysis.fake_percentage,
#                 'real_percentage': latest_analysis.real_percentage,
#                 'frame_count': latest_analysis.frame_count,
#                 'avg_fake_frames': latest_analysis.avg_fake_frames,
#                 'avg_real_frames': latest_analysis.avg_real_frames,
#                 'video_url': '/media/' + latest_analysis.video_path.split('media/')[
#                     1] if 'media/' in latest_analysis.video_path else latest_analysis.video_path,
#                 'created_at': latest_analysis.created_at,
#                 'metadata': metadata,
#                 # Add plot data if it's stored in the model
#                 'plot_data': getattr(latest_analysis, 'plot_data', None)
#             }
#         }
#
#         return render(request, 'temp-dashboard.html', context)
#
#     except VideoAnalysis.DoesNotExist:
#         # If no analysis exists, redirect to home page or appropriate page
#         messages.warning(request, "Please upload a video for analysis first.")
#         return redirect('dashboard')  # Change to an appropriate URL

# def temp_dashboard(request):
#     # Only check if user is authenticated
#     if not request.user.is_authenticated:
#         return redirect('login')
#
#     try:
#         latest_analysis = VideoAnalysis.objects.filter(user=request.user).latest('created_at')
#
#         # Parse the metadata from the stored JSON
#         try:
#             metadata = json.loads(latest_analysis.metadata_json)
#         except (json.JSONDecodeError, AttributeError):
#             metadata = {}
#
#         context = {
#             'analysis': {
#                 'user': latest_analysis.user,
#                 'prediction': latest_analysis.prediction,
#                 'confidence': latest_analysis.confidence,
#                 'fake_percentage': latest_analysis.fake_percentage,
#                 'real_percentage': latest_analysis.real_percentage,
#                 'frame_count': latest_analysis.frame_count,
#                 'avg_fake_frames': latest_analysis.avg_fake_frames,
#                 'avg_real_frames': latest_analysis.avg_real_frames,
#                 'video_url': '/media/' + latest_analysis.video_path.split('media/')[
#                     1] if 'media/' in latest_analysis.video_path else latest_analysis.video_path,
#                 'created_at': latest_analysis.created_at,
#                 'metadata': metadata,
#                 'plot_data': getattr(latest_analysis, 'plot_data', None)
#             },
#             'has_analysis': True
#         }
#
#     except VideoAnalysis.DoesNotExist:
#         # If no analysis exists, still render the template but with a flag indicating no analysis
#         context = {
#             'has_analysis': False,
#             'message': "No video analysis found. Please upload a video for analysis."
#         }
#
#     return render(request, 'temp-dashboard.html', context)

def temp_dashboard(request):
    # Only check if user is authenticated
    if not request.user.is_authenticated:
        return redirect('login')

    try:
        latest_analysis = VideoAnalysis.objects.filter(user=request.user).latest('created_at')

        # Parse the metadata from the stored JSON
        try:
            metadata = json.loads(latest_analysis.metadata_json)
        except (json.JSONDecodeError, AttributeError):
            metadata = {}

        # Get fake_scores from the stored JSON (if you added the fake_scores_json field)
        fake_scores = []
        try:
            if hasattr(latest_analysis, 'fake_scores_json') and latest_analysis.fake_scores_json:
                fake_scores = json.loads(latest_analysis.fake_scores_json)
        except (json.JSONDecodeError, AttributeError):
            fake_scores = []

        # Create chart_data for easier frontend consumption
        chart_data = {}
        if fake_scores:
            chart_data = {
                'frame_numbers': list(range(1, len(fake_scores) + 1)),
                'fake_scores': fake_scores,
                'threshold': 0.5,
                'avg_score': sum(fake_scores) / len(fake_scores) if fake_scores else 0
            }

        context = {
            'analysis': {
                'user': latest_analysis.user,
                'prediction': latest_analysis.prediction,
                'confidence': latest_analysis.confidence,
                'fake_percentage': latest_analysis.fake_percentage,
                'real_percentage': latest_analysis.real_percentage,
                'frame_count': latest_analysis.frame_count,
                'avg_fake_frames': latest_analysis.avg_fake_frames,
                'avg_real_frames': latest_analysis.avg_real_frames,
                'video_url': '/media/' + latest_analysis.video_path.split('media/')[
                    1] if 'media/' in latest_analysis.video_path else latest_analysis.video_path,
                'created_at': latest_analysis.created_at,
                'metadata': metadata,
                'plot_data': getattr(latest_analysis, 'plot_data', None),  # Fixed: Added missing comma
                'fake_scores': fake_scores  # Added fake_scores to analysis object
            },
            'fake_scores': fake_scores,  # Also available at root level
            'chart_data': chart_data,    # Chart-ready data
            'has_analysis': True
        }

    except VideoAnalysis.DoesNotExist:
        # If no analysis exists, still render the template but with a flag indicating no analysis
        context = {
            'has_analysis': False,
            'message': "No video analysis found. Please upload a video for analysis."
        }

    return render(request, 'temp-dashboard.html', context)

def blogs(request):
    return render(request,'blogs.html')


from django.shortcuts import render
from django.core.mail import send_mail
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import json

import logging
from django.core.mail import send_mail
from django.http import HttpResponse

logger = logging.getLogger(__name__)


def test_email(request):
    """
    Test view to check if email sending works
    """
    recipient = request.GET.get('email', 'your-test-email@example.com')

    try:
        logger.info(f"Attempting to send test email to {recipient}")

        # Send test email
        result = send_mail(
            subject='Test Email from Django',
            message='This is a test email to verify SMTP settings.',
            from_email=None,  # Uses DEFAULT_FROM_EMAIL from settings
            recipient_list=[recipient],
            fail_silently=False,
        )

        logger.info(f"Email sending attempt completed with result: {result}")

        if result == 1:
            return HttpResponse(f"Test email sent successfully to {recipient}")
        else:
            return HttpResponse(f"Failed to send test email to {recipient}")

    except Exception as e:
        logger.error(f"Exception while sending email: {str(e)}", exc_info=True)
        return HttpResponse(f"Error sending email: {str(e)}")


import logging
from django.core.mail import send_mail, BadHeaderError
from django.http import JsonResponse, HttpResponseServerError
from django.conf import settings
import json

logger = logging.getLogger(__name__)


def process_contact_form(request):
    """
    View to handle contact form submission with detailed logging and improved field handling
    """
    if request.method == 'POST':
        try:
            # Get form data with consistent field names
            if request.headers.get('Content-Type') == 'application/json':
                data = json.loads(request.body)
                name = data.get('name2')
                email = data.get('email2')
                query = data.get('query')
            else:
                name = request.POST.get('name2')
                email = request.POST.get('email2')
                query = request.POST.get('query')

            # Log the received data
            logger.info(f"Received form data - Name: {name}, Email: {email}")

            # Validate form data
            if not all([name, email, query]):
                missing_fields = []
                if not name:
                    missing_fields.append('name')
                if not email:
                    missing_fields.append('email')
                if not query:
                    missing_fields.append('query')

                logger.warning(f"Form validation failed: missing fields: {', '.join(missing_fields)}")
                return JsonResponse({
                    'status': 'error',
                    'message': 'All fields are required. Please fill in all fields and try again.'
                }, status=400)

            # Basic email validation
            if not '@' in email or not '.' in email:
                logger.warning(f"Form validation failed: invalid email format: {email}")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Please enter a valid email address'
                }, status=400)

            try:
                # Send thank you email to the user
                logger.info(f"Attempting to send thank you email to {email}")

                user_mail_result = send_mail(
                    subject='Thank you for contacting us',
                    message=f'Dear {name},\n\nThank you for choosing us. We care for you and your family. We will contact you soon.\n\nBest regards,\nThe Team',
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[email],
                    fail_silently=False,
                )

                logger.info(f"Thank you email result: {user_mail_result}")

                # Optional: Send notification to admin if configured
                admin_email = getattr(settings, 'ADMIN_EMAIL', None)
                if admin_email:
                    logger.info(f"Attempting to send admin notification to {admin_email}")

                    admin_mail_result = send_mail(
                        subject='New Contact Form Submission',
                        message=f'Name: {name}\nEmail: {email}\nQuery: {query}',
                        from_email=settings.DEFAULT_FROM_EMAIL,
                        recipient_list=[admin_email],
                        fail_silently=False,
                    )

                    logger.info(f"Admin notification email result: {admin_mail_result}")

                return JsonResponse({
                    'status': 'success',
                    'message': 'Thank you! Your message has been sent. We will contact you soon.'
                })

            except BadHeaderError:
                logger.error("Invalid header found in email")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid email header. Please check your email address and try again.'
                }, status=500)

            except Exception as e:
                logger.error(f"Error sending email: {str(e)}", exc_info=True)
                return JsonResponse({
                    'status': 'error',
                    'message': 'Error sending email. Please try again later or contact us directly.'
                }, status=500)

        except json.JSONDecodeError:
            logger.error("Invalid JSON data received")
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid form data received. Please try again.'
            }, status=400)

        except Exception as e:
            logger.error(f"Unexpected error processing form: {str(e)}", exc_info=True)
            return JsonResponse({
                'status': 'error',
                'message': 'An unexpected error occurred. Please try again later.'
            }, status=500)

    # If GET request or any other method
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method. Please submit using the contact form.'
    }, status=405)


def use_case(request):
    return render(request,'use_case.html')
