import requests
import numpy as np
import cv2
import untangle
import os

# The API template for pulls is given by Safebooru https://safebooru.org/index.php?page=help&topic=dapi
# /index.php?page=dapi&s=post&q=index

maxsize = 512
imagecounter = 1
maxImages = 10000
pagestepper = 0
pageoffset = 1
tags = 'magic'
savepath = 'images'

if not os.path.exists(savepath):
    os.makedirs(savepath)

while imagecounter < maxImages:
    try:
        # Get a tagged page
        response = requests.get(
            "http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=" +
            tags + "&pid=" + str(pageoffset + pagestepper))
        response.raise_for_status()
        safebooruXMLPage = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching XML page: {e}")
        break
    
    pagestepper += 1

    try:
        xmlreturn = untangle.parse(safebooruXMLPage)
    except Exception as e:
        print(f"Error parsing XML: {e}")
        continue

    for post in xmlreturn.posts.post:
        imgurl = post["sample_url"]
        if "png" in imgurl or "jpg" in imgurl:
            try:
                resp = requests.get(imgurl)
                resp.raise_for_status()
                image = np.asarray(bytearray(resp.content), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            except requests.exceptions.RequestException as e:
                print(f"Error fetching image: {e}")
                continue
            except Exception as e:
                print(f"Error decoding image: {e}")
                continue

            if image is None:
                print(f"Image not found at URL: {imgurl}")
                continue

            print('counter: {}. URL: {}'.format(imagecounter, imgurl))

            height, width = image.shape[:2]
            
            if height > width:
                scalefactor = (maxsize * 1.0) / width
                res = cv2.resize(
                    image,
                    (int(width * scalefactor), int(height * scalefactor)),
                    interpolation=cv2.INTER_CUBIC)
                cropped = res[0:maxsize, 0:maxsize]
            else:
                scalefactor = (maxsize * 1.0) / height
                res = cv2.resize(
                    image,
                    (int(width * scalefactor), int(height * scalefactor)),
                    interpolation=cv2.INTER_CUBIC)
                center_x = int(round(width * scalefactor * 0.5))
                cropped = res[0:maxsize,
                              center_x - maxsize // 2:center_x + maxsize // 2]

            try:
                cv2.imwrite(
                    os.path.join(savepath, f'{imagecounter}_page_{pagestepper + pageoffset}.jpg'),
                    cropped)
            except Exception as e:
                print(f"Error saving image: {e}")
                continue

            if imagecounter == maxImages:
                break
            else:
                imagecounter += 1

    print(f"Finished processing page {pagestepper + pageoffset}")
    
print("Completed downloading images.")