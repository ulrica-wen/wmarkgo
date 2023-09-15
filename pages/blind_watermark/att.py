# coding=utf-8

# attack on the watermark
import cv2
import numpy as np
import warnings


def cut_att3(input_filename=None, input_img=None, output_file_name=None, loc_r=None, loc=None, scale=None):
    # 剪切攻击 + 缩放攻击
    if input_filename:
        input_img = cv2.imread(input_filename)

    if loc is None:
        h, w, _ = input_img.shape
        x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])
    else:
        x1, y1, x2, y2 = loc

    # 剪切攻击
    output_img = input_img[y1:y2, x1:x2].copy()

    # 如果缩放攻击
    if scale and scale != 1:
        h, w, _ = output_img.shape
        output_img = cv2.resize(output_img, dsize=(round(w * scale), round(h * scale)))
    else:
        output_img = output_img

    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img


cut_att2 = cut_att3


def resize_att(input_filename=None, input_img=None, output_file_name=None, out_shape=(500, 500)):
    # 缩放攻击：因为攻击和还原都是缩放，所以攻击和还原都调用这个函数
    if input_filename:
        input_img = cv2.imread(input_filename)
    output_img = cv2.resize(input_img, dsize=out_shape)
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img


def bright_att(input_filename=None, input_img=None, output_file_name=None, ratio=0.8):
    # 亮度调整攻击，ratio应当多于0
    # ratio>1是调得更亮，ratio<1是亮度更暗
    if input_filename:
        input_img = cv2.imread(input_filename)
    output_img = input_img * ratio
    output_img[output_img > 255] = 255
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img


def shelter_att(input_filename=None, input_img=None, output_file_name=None, ratio=0.1, n=3):
    # 遮挡攻击：遮挡图像中的一部分
    # n个遮挡块
    # 每个遮挡块所占比例为ratio
    if input_filename:
        output_img = cv2.imread(input_filename)
    else:
        output_img = input_img.copy()
    input_img_shape = output_img.shape

    for i in range(n):
        tmp = np.random.rand() * (1 - ratio)  # 随机选择一个地方，1-ratio是为了防止溢出
        start_height, end_height = int(tmp * input_img_shape[0]), int((tmp + ratio) * input_img_shape[0])
        tmp = np.random.rand() * (1 - ratio)
        start_width, end_width = int(tmp * input_img_shape[1]), int((tmp + ratio) * input_img_shape[1])

        output_img[start_height:end_height, start_width:end_width, :] = 255

    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img


def salt_pepper_att(input_filename=None, input_img=None, output_file_name=None, ratio=0.01):
    # 椒盐攻击
    if input_filename:
        input_img = cv2.imread(input_filename)
    input_img_shape = input_img.shape
    output_img = input_img.copy()
    for i in range(input_img_shape[0]):
        for j in range(input_img_shape[1]):
            if np.random.rand() < ratio:
                output_img[i, j, :] = 255
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img


def rot_att(input_filename=None, input_img=None, output_file_name=None, angle=45):
    # 旋转攻击
    if input_filename:
        input_img = cv2.imread(input_filename)
    rows, cols, _ = input_img.shape
    M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
    output_img = cv2.warpAffine(input_img, M, (cols, rows))
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img


def cut_att_height(input_filename=None, input_img=None, output_file_name=None, ratio=0.8):
    warnings.warn('will be deprecated in the future, use att.cut_att2 instead')
    # 纵向剪切攻击
    if input_filename:
        input_img = cv2.imread(input_filename)
    input_img_shape = input_img.shape
    height = int(input_img_shape[0] * ratio)

    output_img = input_img[:height, :, :]
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img


def cut_att_width(input_filename=None, input_img=None, output_file_name=None, ratio=0.8):
    warnings.warn('will be deprecated in the future, use att.cut_att2 instead')
    # 横向裁剪攻击
    if input_filename:
        input_img = cv2.imread(input_filename)
    input_img_shape = input_img.shape
    width = int(input_img_shape[1] * ratio)

    output_img = input_img[:, :width, :]
    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img


def cut_att(input_filename=None, output_file_name=None, input_img=None, loc=((0.3, 0.1), (0.7, 0.9)), resize=0.6):
    warnings.warn('will be deprecated in the future, use att.cut_att2 instead')
    # 截屏攻击 = 裁剪攻击 + 缩放攻击 + 知道攻击参数（按照参数还原）
    # 裁剪攻击：其它部分都补0
    if input_filename:
        input_img = cv2.imread(input_filename)

    output_img = input_img.copy()
    shape = output_img.shape
    x1, y1, x2, y2 = shape[0] * loc[0][0], shape[1] * loc[0][1], shape[0] * loc[1][0], shape[1] * loc[1][1]
    output_img[:int(x1), :] = 255
    output_img[int(x2):, :] = 255
    output_img[:, :int(y1)] = 255
    output_img[:, int(y2):] = 255

    if resize is not None:
        # 缩放一次，然后还原
        output_img = cv2.resize(output_img,
                                dsize=(int(shape[1] * resize), int(shape[0] * resize))
                                )

        output_img = cv2.resize(output_img, dsize=(int(shape[1]), int(shape[0])))

    if output_file_name is not None:
        cv2.imwrite(output_file_name, output_img)
    return output_img


# def cut_att2(input_filename=None, input_img=None, output_file_name=None, loc_r=((0.3, 0.1), (0.9, 0.9)), scale=1.1):
#     # 截屏攻击 = 剪切攻击 + 缩放攻击 + 不知道攻击参数
#     if input_filename:
#         input_img = cv2.imread(input_filename)
#     h, w, _ = input_img.shape
#     x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])
#
#     output_img = cut_att3(input_img=input_img, output_file_name=output_file_name,
#                           loc=(x1, y1, x2, y2), scale=scale)
#     return output_img, (x1, y1, x2, y2)

def anti_cut_att_old(input_filename, output_file_name, origin_shape):
    warnings.warn('will be deprecated in the future')
    # 反裁剪攻击：复制一块范围，然后补全
    # origin_shape 分辨率与约定理解的是颠倒的，约定的是列数*行数
    input_img = cv2.imread(input_filename)
    output_img = input_img.copy()
    output_img_shape = output_img.shape
    if output_img_shape[0] > origin_shape[0] or output_img_shape[0] > origin_shape[0]:
        print('裁剪打击后的图片，不可能比原始图片大，检查一下')
        return

    # 还原纵向打击
    while output_img_shape[0] < origin_shape[0]:
        output_img = np.concatenate([output_img, output_img[:origin_shape[0] - output_img_shape[0], :, :]], axis=0)
        output_img_shape = output_img.shape
    while output_img_shape[1] < origin_shape[1]:
        output_img = np.concatenate([output_img, output_img[:, :origin_shape[1] - output_img_shape[1], :]], axis=1)
        output_img_shape = output_img.shape

    cv2.imwrite(output_file_name, output_img)


def anti_cut_att(input_filename=None, input_img=None, output_file_name=None, origin_shape=None):
    warnings.warn('will be deprecated in the future, use att.cut_att2 instead')
    # 反裁剪攻击：补0
    # origin_shape 分辨率与约定理解的是颠倒的，约定的是列数*行数
    if input_filename:
        input_img = cv2.imread(input_filename)
    output_img = input_img.copy()
    output_img_shape = output_img.shape
    if output_img_shape[0] > origin_shape[0] or output_img_shape[0] > origin_shape[0]:
        print('裁剪打击后的图片，不可能比原始图片大，检查一下')
        return

    # 还原纵向打击
    if output_img_shape[0] < origin_shape[0]:
        output_img = np.concatenate(
            [output_img, 255 * np.ones((origin_shape[0] - output_img_shape[0], output_img_shape[1], 3))]
            , axis=0)
        output_img_shape = output_img.shape

    if output_img_shape[1] < origin_shape[1]:
        output_img = np.concatenate(
            [output_img, 255 * np.ones((output_img_shape[0], origin_shape[1] - output_img_shape[1], 3))]
            , axis=1)

    if output_file_name:
        cv2.imwrite(output_file_name, output_img)
    return output_img

def fft_att(input_filename=None, input_img=None, output_file_name=None):
    if input_filename:
        img = cv2.imread(input_filename)
        Shape = img.shape[:2]
        input_img = cv2.imread(input_filename)[:,:,0]
    
    Cropshape = min(Shape[0], Shape[1])
    img = input_img[:Cropshape, :Cropshape] # crop to 600 x 600 
    r = 50 # how narrower the window is
    ham = np.hamming(Cropshape)[:,None] # 1D hamming
    ham2d = np.sqrt(np.dot(ham, ham.T)) ** r # expand to 2D hamming
    # ham2d

    f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = ham2d * f_complex

    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.abs(inv_img)
    minn = filtered_img.min()
    maxx = filtered_img.max()
    filtered_img -= minn
    filtered_img = filtered_img*255 / maxx
    filtered_img = filtered_img.astype(np.uint8)
    if output_file_name:
        cv2.imwrite(output_file_name, filtered_img)
    return filtered_img, minn, maxx, Cropshape


def fft_att2(input_filename=None, image=None, output_file_name=None, threshold = 25):
    if input_filename:
        image = cv2.imread(input_filename)
    img_shape = image.shape
    shape = (img_shape[0], img_shape[1], 3)
    input_img = (image[:,:,0]).astype(np.float32) # /255
    # input_img = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
    color1 = image[:,:,1].astype(np.float32)
    color2 = image[:,:,2].astype(np.float32)
    # Scaling is not a problem

    ## Convert the image from spatial domain to the frequency domain using discrete fourier transform
    fft = cv2.dft(input_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    # Shifting the return of the fft so the low frequency will be in the center of the array
    fft_shift = np.fft.fftshift(fft, axes=[0, 1])
    # Manipulate the high frequency
    mask = np.zeros(fft_shift.shape, np.uint8)
    mask[:] = 100
    mask[mask.shape[0]//2-threshold:mask.shape[0]//2+threshold,
        mask.shape[1]//2-threshold:mask.shape[1]//2+threshold, :] = 1
    fft_shift *= mask
    # Shift back
    fft = np.fft.ifftshift(fft_shift, axes=[0, 1])
    # Convert from frequency domain into spatial domain using inverse fft
    filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    if output_file_name:
        cv2.imwrite(output_file_name, filtered)
    df = np.zeros(shape)
    df[:, :, 2] = color2.astype(np.float32)
    df[:, :, 1] = color1.astype(np.float32)
    df[:, :, 0] = filtered
    return filtered, df


def anti_fft_att(input_filename = None, input_img = None,  output_file_name = None, Cropshape = None, maxx = None, minn = None):
    if input_filename:
        img = cv2.imread(input_filename)
        # input_img = cv2.imread(input_filename)[:,:,0]
    # img = input_img[:600,:600]
    original_img = img * maxx / 255
    original_img += minn
    ################################Trial without reversing the abs function#########################
    
    inv_img = cv2.dft(original_img.astype(np.float32), flags = cv2.DFT_COMPLEX_OUTPUT)
    img_shift = np.fft.fftshift(inv_img)
    
    
    r = 50 # how narrower the window is
    ham = np.hamming(Cropshape)[:,None] # 1D hamming
    ham2d = np.sqrt(np.dot(ham, ham.T)) ** r # expand to 2D hamming

    
    img_complex = img_shift[:,:,0]*1j + img_shift[:,:,1]
    original_img = img_complex / ham2d
    original_img_shifted = np.fft.fftshift(original_img)
    original_img_reversed = np.fft.ifft2(original_img_shifted)
    final_img = np.abs(original_img_reversed)
    final_img = final_img.astype(np.uint8)
    if output_file_name:
        cv2.imwrite(output_file_name, final_img)
    return final_img
    ################################Trial with reversing the abs function############################
    # Search for maxx and minn function
    
    
def anti_fft_att2(input_filename = None, input_img = None,  output_file_name = None, color1 = None, color2 = None, threshold = 25):
    if input_filename:
        input_img = cv2.imread(input_filename)[:,:,0].astype(np.float32)
    
    img_shape = input_img.shape
    shape = (img_shape[0], img_shape[1], 3)
    fft = cv2.dft(input_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_shift = np.fft.fftshift(fft, axes=[0, 1])
    sz = threshold
    mask = np.zeros(fft_shift.shape, np.uint8)
    mask[:] = 100
    mask[mask.shape[0]//2-sz:mask.shape[0]//2+sz,
        mask.shape[1]//2-sz:mask.shape[1]//2+sz, :] = 1
    fft_shift /= mask
    fft = np.fft.ifftshift(fft_shift, axes=[0, 1])
    filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    df = np.zeros(shape)
    df[:,:,2] = color2
    df[:,:,1] = color1
    df[:,:,0] = filtered
    if output_file_name:
        cv2.imwrite(output_file_name, df)
    return df

def estimate_para_fft(input_filename = None, input_img = None,  output_file_name = None, maxx = None, minn = None):
    if maxx and minn:
        return maxx, minn
    if input_filename:
        img_est = cv2.imread(input_filename)[:,:,0]
    

