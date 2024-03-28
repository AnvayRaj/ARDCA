import subprocess
import argparse
import datetime
import os
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from flask import Flask, render_template

def open_file(file_path):
    try:
        subprocess.Popen(['xdg-open', file_path])  # For Linux
    except OSError:
        try:
            subprocess.Popen(['open', file_path])  # For macOS
        except OSError:
            subprocess.Popen(['start', '', file_path], shell=True)  # For Windows

def pareto_chart(data):
    # Sort the dictionary by values in descending order
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

    # Calculate cumulative frequencies and total frequency
    total = sum(sorted_data.values())
    cumulative_percentage = 0
    cumulative_frequencies = []
    for value in sorted_data.values():
        cumulative_percentage += (value / total) * 100
        cumulative_frequencies.append(cumulative_percentage)

    # Create the Pareto chart
    fig,ax1 = plt.subplots()

    ax1.bar(sorted_data.keys(), sorted_data.values(), color='b')
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Frequency')

    plt.title('Pareto Distribution')
    plt.xticks(ha='center')
    plt.savefig("./static/pareto.jpg")
    # plt.show()
    return sorted_data


def read_images_sorted(directory):
  images = []
  # Get filenames sorted in ascending order
  filenames = sorted(os.listdir(directory))
  for filename in filenames:
    # Check if it's an image file
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
      # Construct the full path
      filepath = os.path.join(directory, filename)
      # Read the image with cv2
      image = cv2.imread(filepath)
      if image is not None:
        images.append(image)
  return images

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Sampling and Analyzing video frames.')

    # Add arguments
    parser.add_argument('-vf', '--videofile', help='Path to the input video file', required=True)

    # Parse arguments
    args = parser.parse_args()

    # Access parsed arguments
    video_file_path = args.videofile

    # ----- Final Query Execution -----
    query = """./Utilities/ffmpeg.exe -i input.mp4 -vf fps=1 -loglevel quiet -stats ./Output/output_%04d.png"""
    subprocess.run(query)

    # ----- Process Frames -----
    directory = "./Output"
    images = read_images_sorted(directory)

    model = load_model('model_file.h5')
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    raw_test_results = []

    if images:
        # Processing Frames
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 3)
            for x, y, w, h in faces:
                sub_face_img = gray[y:y+h, x:x+w]
                resized = cv2.resize(sub_face_img, (48, 48))
                normalize = resized/255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                raw_test_results.append(label)
    else:
        print("No images found in the directory!")

    duration_in_seconds = len(images)

    # ----- Remove Temporary Files -----
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)

    # print(raw_test_results)

    # ----- Plot Results -----
    color_map = {0: 'red', 1: 'green', 2: 'purple', 3: 'orange', 4: 'grey', 5: 'blue', 6: 'yellow'}

    # Create an empty plot with labeled axes
    plt.figure(figsize=(8, 2))
    plt.xlabel("Time (sec)")
    plt.title("Timeseries plot of emotion classes")

    # Set y-axis limits (assuming values don't go beyond 0 and 1)
    plt.ylim(0, 1)

    # Starting time (adjust if needed)
    start_time = 0

    # Loop through the data and create axvspan rectangles
    for value in raw_test_results:
        end_time = start_time + 1  # Assuming each value represents a unit of time
        plt.axvspan(start_time, end_time, color=color_map[value], alpha=1)  # Adjust alpha for transparency
        start_time = end_time

    # plt.grid(True)  # Add gridlines
    legend_elements = [Patch(facecolor=color,label=labels_dict[idx]) for idx,color in color_map.items()]

    plt.legend(handles=legend_elements, bbox_to_anchor = (1.25, 0.6), loc='center left')
    plt.tight_layout()
    plt.savefig("./static/axvspan.jpg")
    frequency_count = dict()
    for cls in labels_dict.values():
       frequency_count[cls] = 0
    for cls in raw_test_results:
       frequency_count[labels_dict[cls]]+=1
    pareto=pareto_chart(frequency_count)
    print("Plot Saved!")
    # plt.show()

    command = """./Utilities/ffmpeg.exe -i "./Ad Video.mp4" -ss """+str(duration_in_seconds//2)+""" -vframes 1 "./static/thumbnail.jpeg" """
    subprocess.run(command, check=True)

    info = {
       "video_title": "Ad Video",
       "duration": duration_in_seconds,
       "emotional_tone": list(pareto.keys())[:3],
       "engagement_score": round((duration_in_seconds-pareto["Neutral"])/duration_in_seconds,2),
       "time_stamp": str(datetime.datetime.now())
    }
    return info





if __name__ == '__main__':
    info = main()

    app = Flask(__name__)

    @app.route('/')
    def index():
        # Render the HTML template
        return render_template('index.html', info=info)
    
    app.run()
    
    
