from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ProcessedVideo
from rest_framework.decorators import api_view
import os
from .visualize import func


@api_view(['POST'])
@csrf_exempt
def process_video(request):
    if request.method == 'POST':
        # Assuming the video file is sent as form data with key 'videofile'
        video_file_path = request.data.get('videofile')

        print(video_file_path)
        
        # Process the video here using your ML model
        info = func(video_file_path)

        # Save the processed information to the database
        processed_video = ProcessedVideo.objects.create(
            video_file = video_file_path, 
            video_title = info.get('video_title'),
            duration = info.get('duration'),
            emotional_tone = info.get('emotional_tone'),
            engagement_score = info.get('engagement_score'),
            time_stamp = info.get('time_stamp')
        )

        # Delete the video file after processing
        if os.path.exists(video_file_path):
            os.remove(video_file_path)

        # Return the processed information
        return JsonResponse(info)
        # return JsonResponse({'msg': 'sike'})

    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
