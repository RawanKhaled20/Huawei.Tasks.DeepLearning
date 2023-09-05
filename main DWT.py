import cv2
from PIL import Image
import numpy as np
from scipy import fft
import pywt
import pywt.data
import os
import time

class DWT:
    def encode_img(self, img, message, encoded_img_name):
        img = Image.open(img, 'r')
        # c, r = img.size # c - columns (width), r - rows (height)
        message_bits = np.asanyarray([ord(c) for c in message], dtype='uint8')  # converting message to binary
        message_bits = np.unpackbits(message_bits)

        pixel_values = np.array(img)
        R = pixel_values[:, :, 0]
        G = pixel_values[:, :, 1]
        B = pixel_values[:, :, 2]

        encoded_pixels_2 = []

        for x in range(0, len(R)):
            encoded_pixels_2.append(np.column_stack((R[x], G[x], B[x])))

        # 2D Haar transform on BLUE CHANNEL!!
        coeffs2 = pywt.dwt2(B, 'haar')
        LL, (LH, HL, HH) = coeffs2

        if (len(HH) < len(message_bits)):
            print("Message too long!")
            return

        HH_uint8 = HH.astype(np.uint8)
        HH_bits = []
        message_index = 0

        # HH converstion from float to binary: float32 -> uint8 -> binarny
        # last bit from  HH_uint8[x] is taken and message bits are inserted
        for x in range(0, len(HH_uint8)):
            bits = []
            if (message_index == len(message_bits)):
                break
            for n in range(0, len(HH_uint8[x])):
                bits.append(np.unpackbits(HH_uint8[x][n]))

            HH_bits.append(bits)

        tmp = []
        for i in range(1, len(HH_bits)):
            HH_bits[i][-1][0] = message_bits[message_index]

            message_index += 1

            if (message_index == len(message_bits)): break

        HH_uint8_converted = []

        for i in range(0, len(HH_bits)):
            val = []
            for j in range(0, len(HH_bits[i])):
                val.append(np.packbits(HH_bits[i][j])[0])
            HH_uint8_converted.append(val)

        HH_float = np.float32(HH_uint8_converted)
        # HH_float[-1] = np.float32([255] * 256)

        coeffs_encoded = LL, (LH, HL, HH_float)
        B_encoded = pywt.idwt2(coeffs_encoded, 'haar') // 1

        encoded_pixels = []

        for x in range(0, len(R)):
            tmp = []
            for y in range(0, len(R[x])):
                pixel = (R[x][y], G[x][y], B_encoded[x][y])
                tmp.append(pixel)
            encoded_pixels.append(tmp)

        encoded_pixels = np.asarray(encoded_pixels, dtype='uint8')
        img_encoded = Image.fromarray(encoded_pixels)
        img_encoded.save(encoded_img_name)

    def decode_img(self, img):
        img = Image.open(img, 'r')
        pixel_values = np.array(img)
        R = pixel_values[:, :, 0]
        G = pixel_values[:, :, 1]
        B = pixel_values[:, :, 2]

        coeffs2 = pywt.dwt2(B, 'haar')
        LL, (LH, HL, HH) = coeffs2

        HH_uint8 = HH.astype(np.uint8)
        last_bits_1 = []

        for x in range(0, len(HH_uint8)):
            for n in range(0, len(HH_uint8[x])):
                last_bits_1.append(format(HH_uint8[x][n], '08b')[0])

        bits_1 = ''.join(last_bits_1)

        chars_1 = str(([chr(int('0b' + bits_1[i:i + 7], 2)) for i in range(0, len(bits_1), 7)][:300]))

        return chars_1

def main(ifile, ofile , secret_msg,time_list):
    start_time = time.time()
    dwt_obj = DWT()
    img_png=ifile
    dwt_obj.encode_img(img_png, secret_msg, ofile)
    end_time = time.time()
    #dwt_msg = dwt_obj.decode_img(ofile)
    #print(dwt_msg)
    # Calculate the execution time for this image
    exec_time = end_time - start_time

    # Add the execution time to the list
    time_list.append(exec_time)


  # Loop through each image file in the folder
def calculate(folder,output, mse_list ,psnr_list):
    for file in os.listdir(folder):
          # Construct the full file paths for the input and output images
          input_file_path = os.path.join(folder, file)
          output_file_path = os.path.join(output, file)
          # Load the input and output images
          img_in = cv2.imread(input_file_path)
          img_out = cv2.imread(output_file_path)
          # Calculate the MSE between the two images
          mse = np.mean((img_in - img_out) ** 2)

          # Calculate the maximum pixel value
          max_pixel = 255.0  # for 8-bit grayscale images

          # Calculate the PSNR between the two images (for steganography)
          if mse == 0:
              psnr = 100
          else:
              psnr = 10 * np.log10((max_pixel ** 2) / mse)

          # Add the MSE and PSNR values to the arrays
          mse_list.append(mse)
          psnr_list.append(psnr)

folder = 'in'
output = 'out'
secret_msg = "secretmessage$$"
time_list =[]
mse_list=[]
psnr_list=[]
for file in os.listdir(folder):
  main(f"{folder}/{file}", f"{output}/{file}",secret_msg,time_list)
avg_time = np.mean(time_list)
print('Average execution time:', avg_time)
calculate(folder,output,mse_list,psnr_list)
avg_mse = np.mean(mse_list)
avg_psnr = np.mean(psnr_list)

# Print the average MSE, PSNR, and execution time
print('Average MSE:', avg_mse)
print('Average PSNR:', avg_psnr)

