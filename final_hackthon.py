# Import necessary libraries
from twilio.rest import Client
import cv2
import datetime
import pyttsx3
import wikipedia
import torch
import torchvision.transforms as transforms

# Replace these values with your Twilio account SID, auth token, and Twilio phone number
TWILIO_ACCOUNT_SID = 'AC63f6fb9b1e1543ac5666d60706960402'
TWILIO_AUTH_TOKEN = '7e1ea3414c8cdcfb58f8fdef80e6898e'
TWILIO_PHONE_NUMBER = '+16614491199'

# Dictionary of contacts with their phone numbers
contacts = {
    'bhanodai': '+918374613658',
    'ram': '+919994318593',
    'charwick': '8008846040'
}

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load YOLO classes
yolo_classes_file = 'C:\\Users\\Ramana\\Downloads\\coco.names'  # Replace with the path to your classes file
with open(yolo_classes_file, 'r') as f:
    yolo_classes = [line.strip() for line in f.readlines()]

# Load YOLOv3 model for object detection
yolo_net = cv2.dnn.readNet('C:\\Users\\Ramana\\Downloads\\yolov3.weights', 'C:\\Users\\Ramana\\Downloads\\yolov3.cfg')

# Print information about YOLOv3 network
print("YOLOv3 network information:")
output_layers = yolo_net.getUnconnectedOutLayersNames()

for i, output_layer in enumerate(output_layers):
    print(f"{i + 1}: {output_layer}")

# Load MiDAS model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval()

# Define the input transformation pipeline for MiDAS manually
transform_midas = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
])

# Constants for danger depth threshold
danger_depth_threshold = 600

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def get_user_input(prompt):
    print(prompt)
    return input().lower()

def send_sms():
    # Ask for contact name
    contact_name = get_user_input("To whom should I send the message?")
    if not contact_name:
        print("Please try again.")
        return

    # Ask for the message
    message = get_user_input("What message would you like to send?")
    if not message:
        print("Please try again.")
        return

    # Send SMS
    if contact_name in contacts:
        to_phone_number = contacts[contact_name]
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone_number
        )
        print(f"Message sent to {contact_name} at {to_phone_number}. Message SID: {message.sid}")
        speak("Message sent successfully.")
    else:
        print(f"Contact '{contact_name}' not found.")
        speak(f"Contact {contact_name} not found. Please try again.")

def search_wikipedia():
    # Ask for the search query
    query = get_user_input("What would you like to search on Wikipedia?")
    if not query:
        print("Please try again.")
        return

    # Search Wikipedia
    speak("Searching Wikipedia")
    try:
        results = wikipedia.summary(query)
        speak("According to Wikipedia")
        speak(results)
    except wikipedia.exceptions.DisambiguationError:
        speak(f"Multiple results found for {query}. Please be more specific.")
    except wikipedia.exceptions.PageError:
        speak(f"No results found for {query}. Please try another query.")

def object_detection():
    cap = cv2.VideoCapture('http://192.168.137.26:4747/video')
    while cap.isOpened():
        ret, frame = cap.read()

        # Check if the image has an alpha channel (4 channels)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Remove alpha channel

        # Perform object detection using YOLOv3
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        yolo_outs = yolo_net.forward(output_layers)

        # Extract class IDs and bounding boxes from YOLO output
        class_ids = []
        confidences = []
        boxes = []
        for out in yolo_outs:
            for detection in out:
                scores = torch.tensor(detection[5:])
                class_id = torch.argmax(scores).item()
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Perform inference for each object bounding box using MiDAS
        for box in boxes:
            x, y, w, h = box
            object_region = frame[y:y + h, x:x + w]

            # Apply the transformation to the object region for MiDAS
            input_tensor = transform_midas(object_region)
            if input_tensor.dim() == 3 and input_tensor.size(0) > 0 and input_tensor.size(1) > 0:
                input_tensor = input_tensor.unsqueeze(0)

                # Perform depth estimation using MiDAS
                with torch.no_grad():
                    prediction = midas(input_tensor)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(h, w),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze()

                    # Display the depth map for the object (assuming it's a single-channel tensor)
                    depth_map_object = prediction.squeeze().cpu().numpy()
                    cv2.imshow('Depth Map Object', depth_map_object)

                    # Check for dangerous proximity based on the depth map
                    depth_mean = depth_map_object.mean()
                    print("Depth Mean for Object:", depth_mean)

                    # Check if the depth is less than the specified threshold
                    if depth_mean < danger_depth_threshold:
                        print("Dangerous proximity detected for object!")
                        speak("Warning: There is a dangerous object nearby!")
                    else:
                        speak("The path is clear.")

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

def get_time():
    strTime = datetime.datetime.now().strftime("%H:%M")
    speak(f"The time is {strTime}")

def wish():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak('Good morning')
        speak("How can I help you?")
    elif 12 <= hour < 18:
        speak('Good afternoon')
        speak("How can I help you?")
    else:
        speak('Good evening')
        speak("How can I help you?")

def display_options():
    print("Options:")
    print("1. Send a Message")
    print("2. Search Wikipedia")
    print("3. Object Detection")
    print("4. Get the Time")
    print("5. Quit")

def main():
    wish()

    while True:
        display_options()
        command = get_user_input("Enter your choice (1-5): ")

        if '1' in command:
            send_sms()
            speak("The task is done.")
        elif '2' in command:
            search_wikipedia()
        elif '3' in command:
            object_detection()
        elif '4' in command:
            get_time()
        elif '5' in command:
            break
        else:
            speak("I'm sorry, I didn't understand that. Please try again.")

if __name__ == "__main__":
    main()
