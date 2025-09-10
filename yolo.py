import torch
import cv2
import geocoder
import time
from ultralytics import YOLO
from twilio.rest import Client
import ultralytics

account_sid = "AC5a8f96c83310a07c3b6e0378b17e86dc"  
auth_token = "d13409d8255de98890d7b35455896932"    
twilio_phone_number = "+19342259318"  
to_phone_number = "+918763458226"    

client = Client(account_sid, auth_token)


model = YOLO(r"C:\Users\shiva\Desktop\weight\yolov8l.pt", verbose=False)

classNames = model.names  

confidence_threshold = 0.95  
cooldown_period = 30  
last_triggered_time = 0  

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", 1280, 720)

def get_location():
    """Fetch the current location using geocoder."""
    g = geocoder.ip("me")  
    if not g.ok:
        return "📍 Unable to fetch location."
    return f"📍 Location: {g.city}, {g.state}, {g.country} (Lat: {g.lat}, Lng: {g.lng})"

def send_alert(sms_body):
    """Send SMS & WhatsApp Alert"""
    try:
     
        sms_message = client.messages.create(
            body=sms_body,
            from_=twilio_phone_number,
            to=to_phone_number
        )
        print(f"📩 SMS Sent! SID: {sms_message.sid}")

        whatsapp_message = client.messages.create(
            body=sms_body,
            from_="whatsapp:+14155238886",  
            to=f"whatsapp:+918763458226"  
        )
        print(f"📩 WhatsApp Message Sent! SID: {whatsapp_message.sid}")

    except Exception as e:
        print(f"❌ Error sending alerts: {e}")

try:
    while True:
        success, img = cap.read()
        if not success:
            print("❌ Failed to capture frame. Exiting...")
            break  

        results = model(img, stream=True)
        detected = False  

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                confidence_percentage = round(confidence, 2)

                cls = int(box.cls[0])
                detected_class = classNames[cls] if cls < len(classNames) else f"Unknown Class {cls}"
                
        
                print(f"🔍 Detected: {detected_class} | Confidence: {confidence_percentage * 100}%")

                if confidence_percentage > confidence_threshold:
                    detected = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, detected_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    current_time = time.time()
                    if current_time - last_triggered_time >= cooldown_period:
                        print("🚨 High confidence detected. Triggering SOS alert...")

                        location_info = get_location()
                        print(location_info)

                        try:
                            call = client.calls.create(
                                to=to_phone_number,
                                from_=twilio_phone_number,
                                url="https://your-public-url.com/voice.xml" 
                            )
                            print(f"📞 SOS Call Sent! Call SID: {call.sid}")

                            sms_body = f"⚠️ Emergency Alert! {detected_class} detected with {confidence_percentage * 100}% confidence.\n{location_info}"

                            send_alert(sms_body)

                            last_triggered_time = current_time  
                        except Exception as e:
                            print(f"❌ Twilio Error: {e}")

        cv2.imshow("Webcam", img)

        if cv2.waitKey(1) == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
