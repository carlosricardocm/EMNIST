import logging
logging.basicConfig(filename='log.log', level=logging.INFO)
import datetime
import os
import cv2
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import ndimage

import math
from extra_keras_datasets import emnist

import numpy as np
import pandas as pd
import sys

#SMAC for get best params for EAM's
import SMAC_run_sameconfigforall as smac

# EAM
import constants
import convnet
from associative import AssociativeMemory

#For progress bar
from tqdm import tqdm


(train_images, train_labels), (test_images, test_labels) = emnist.load_data(type='balanced')

#Parte de la carga del dataset IAM

iam_sources_path = os.path.join('databases', 'IAM')

# Read lines annotations
array_lines=[]
n=0
with open(os.path.join(iam_sources_path, 'ascii','lines.txt'), 'r') as f:
    array_words = []
    for line in f:
        if line[0] !='#':
            lp = line.strip().split(' ')
            n+=1
            array_lines.append((lp[0], lp[1], int(lp[2]), int(lp[3]), int(lp[4]), int(lp[5]), int(lp[6]), int(lp[7]), ' '.join(lp[8:])))

pd_lines = pd.DataFrame(array_lines, columns=['id_line','segmentation_result','graylevel_binarize', ' num_components',
                                              'x','y','w','h','word'])

pd_lines['id_page'] = pd_lines['id_line'].map(lambda x: '-'.join(x.split('-')[:2]))
pd_lines['id_writter'] = pd_lines['id_page'].map(lambda x: x.split('-')[0])
#pd_lines['partition'] = pd_lines['id_page'].apply(lambda x: partition_dict.get(x, 'trn'))

# Read word annotations
array_words=[]
n=0
with open(os.path.join(iam_sources_path, 'ascii','words.txt'), 'r') as f:
    array_words = []
    for line in f:
        if line[0] !='#':
            lp = line.strip().split(' ')
            n+=1
            array_words.append((lp[0], lp[1], int(lp[2]), int(lp[3]), int(lp[4]), int(lp[5]), int(lp[6]), lp[7], ' '.join(lp[8:])))

pd_words = pd.DataFrame(
    array_words,
    columns=[
        'id_word','segmentation_result','graylevel_binarize',
        'x','y','w','h','grammar_tag','word'
    ]
)

pd_words['line'] = pd_words.id_word.apply(lambda x: '-'.join(x.split('-')[:-1]))
pd_words['page'] = pd_words.line.apply(lambda x: '-'.join(x.split('-')[:-1]))

# marca de las palablas a seleccionar
pd_words['selected'] = False
pd_words.loc[(pd_words.segmentation_result == 'ok') & (pd_words.word != "#") & (pd_words.x > 0), 'selected'] = True

lines_selected = set(pd_words[pd_words.selected].line.values)

notebook_name = 'preprocesamiento'

destination_folder = os.path.join('databases', 'IAM', 'normalizada')
iam_sources_path = os.path.join('databases', 'IAM')
iam_filename = "iamdataset.npz"

test_img_page = os.path.join(iam_sources_path, 'formsA-D', 'a01-000x.png')
test_img_line = os.path.join(iam_sources_path, 'lines', 'a02', 'a02-000','a02-000-00.png')

#logging
know_time = datetime.datetime.now()

log_name = notebook_name+'-'+str(know_time.year)+"_"+str(know_time.month).zfill(2)+"_"+str(know_time.day).zfill(2)\
             +"_"+str(know_time.hour).zfill(2)+"_"+str(know_time.minute).zfill(2)+".log"
if not os.path.exists("logs"):
    os.makedirs("logs")


logger = logging.getLogger(notebook_name)
logging.basicConfig()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr = logging.FileHandler("logs/"+log_name)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(20)
logger.setLevel(logging.DEBUG)
#end loggin

def enhance_contrast(img):
    image = cv2.imread(img, 0)  # read as grayscale
    th = cv2.adaptiveThreshold(image, 
        255,  # maximum value assigned to pixel values exceeding the threshold
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # gaussian weighted sum of neighborhood
        cv2.THRESH_BINARY,  # thresholding type
        5,  # block size (5x5 window)
        3)  # constant
    return th

def enhance_contrast_otsu(img):
    image = cv2.imread(img, 0)  # read as grayscale
    ret, th = cv2.threshold(image,
        0,  # threshold value, ignored when using cv2.THRESH_OTSU
        255,  # maximum value assigned to pixel values exceeding the threshold
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresholding type
    return th


def slope_correction(img):
    img_out = correct_line_inclination(img)
    return img_out

def correct_line_inclination(img):

    #Detect baseline to correct inclination
    
    y0, y1, y_mean, angle = detect_baseline(img)
    logger.debug(f"rotate angle: {angle}")
    
    # Correct inclination with lower angle
    img_out = ndimage.rotate(img, angle)

    img_out = rescale(img_out)
    
    img_out = crop_borders(img_out)
    
    return img_out

def crop_borders(img):
    ''' Crop borders
    '''
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(img)
    if len(true_points)>0:
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0)
        img = img[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
                  top_left[1]:bottom_right[1]+1]  # inclusive
    return img

def rescale(img, threshold=20):
    img[img < 0] = 0
    img2 = np.array((img - np.min(img)) * (255 / (np.max(img)-np.min(img)) ) )
    img2[img2 < threshold] = 0
    return img2

def detect_baseline(img, treshold=20, drawline = False):
    '''
    Para calcular la linea base primer se itera sobre todos los pixeles en las columnas y se guarda el valor donde el pixel pase de negro
    blanco, despues se realiza una regresion lineal para calcular las mejores coordenadas donde pasará la linea base.
    '''
    
    low = []
    #itera sobre las columnas ya que shape[1] nos da el ancho de la imagen
    for w in range(1,img.shape[1]-1):
        #Sobre cada columna buscamos la posición de abajo hacia arriba donde cambio de color negro a blanco y guardamos esa coordenada
        #Adicionalmente checamos que el pixel a buscar pase un umbral de color es decir sea realmente un pixel blanco
        if np.max(img[:,w]) > treshold:
            for h in range(img.shape[0]-5, 0, -1):                
                if img[h,w] > treshold:
                    low += [[h,w]]
                    break
    #Al final obtenemos un arreglo con coordenadas x,y que indican donde esta el pixel con el color blanco mas abajo en cada columna de la imagen
    points_lower = np.array(low)

    #Dibujamos la nube de puntos
    if drawline:
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for item in points_lower:
            cv2.drawMarker(output, (item[1], item[0]),(0,0,255), markerType=cv2.MARKER_STAR, 
            markerSize=1, thickness=2, line_type=cv2.LINE_AA)
        cv2.imshow('clowd point image', output)
        
    
    #Regresion Lineal
    xs = points_lower[:,1]
    ys = points_lower[:,0]
    x = points_lower[:,1].reshape(points_lower.shape[0],1)
    y = points_lower[:,0].reshape(points_lower.shape[0],1)
    #Tal ves de pueda usar un modelo robusto como  RANSACRegressor o Theil-Sen
    model = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model.fit(x, y)
    y0 = model.predict(x[0].reshape(1, -1))[0][0]
    y1 = model.predict(x[-1].reshape(1, -1))[0][0]
    x_mean = img.shape[1]/2
    y_mean = model.predict(np.array([img.shape[1]/2]).reshape(1, -1))
    
    angle = np.arctan((y1 - y0) / (x[-1] - x[0])) * (180 / math.pi)
    
    #Dibujamos la línea obtenida por la regresión lineal y la línea original
    if(drawline):      
        #Agregamos el canal de color extra a la imagen en escala de grises para poder poner de color rojo la linea del baseline
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #Los puntos y0, y_mean y y1 son los puntos calculados por la regresion lineal
        pts = np.array([[x[0], y0], [x_mean, y_mean], [x[-1], y1]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        yo_mean = y[int(img.shape[1]/2)]
        pts_origin = np.array([[x[0], y[0]], [x_mean, yo_mean], [x[-1], y[-1]]], np.int32)
        pts_origin = pts.reshape((-1, 1, 2))               
        cv2.polylines(output, 
              [pts_origin], 
              isClosed = False,
              color = (255,0,0),
              thickness = 1)
        out = output.copy()
        cv2.polylines(out, 
              [pts], 
              isClosed = False,
              color = (0,0,255),
              thickness = 1)
        cv2.imshow('baseline image', out)

    return y0, y1, int(y_mean), angle[0]


def detect_upperlane(img, treshold=20, drawline = False):
    '''
    Para calcular la linea base primer se itera sobre todos los pixeles en las columnas y se guarda el valor donde el pixel pase de negro
    blanco, despues se realiza una regresion lineal para calcular las mejores coordenadas donde pasará la linea base.
    '''
    
    high = []
    #itera sobre las columnas ya que shape[1] nos da el ancho de la imagen
    for w in range(1,img.shape[1]-1):
        #Sobre cada columna buscamos la posición de abajo hacia arriba donde cambio de color negro a blanco y guardamos esa coordenada
        #Adicionalmente checamos que el pixel a buscar pase un umbral de color es decir sea realmente un pixel blanco
        if np.max(img[:,w]) > treshold:
            for h in range(5, img.shape[0]):                
                if img[h,w] > treshold:
                    high += [[h,w]]
                    break
    #Al final obtenemos un arreglo con coordenadas x,y que indican donde esta el pixel con el color blanco mas abajo en cada columna de la imagen
    points_higher = np.array(high)

    #Dibujamos la nube de puntos
    if drawline:
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for item in points_higher:
            cv2.drawMarker(output, (item[1], item[0]),(0,0,255), markerType=cv2.MARKER_STAR, 
            markerSize=1, thickness=2, line_type=cv2.LINE_AA)
        cv2.imshow('clowd point image upper', output)
        
    
    #Regresion Lineal
    xs = points_higher[:,1]
    ys = points_higher[:,0]
    x = points_higher[:,1].reshape(points_higher.shape[0],1)
    y = points_higher[:,0].reshape(points_higher.shape[0],1)
    #Tal ves de pueda usar un modelo robusto como  RANSACRegressor o Theil-Sen
    model = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model.fit(x, y)
    y0 = model.predict(x[0].reshape(1, -1))[0][0]
    y1 = model.predict(x[-1].reshape(1, -1))[0][0]
    x_mean = img.shape[1]/2
    y_mean = model.predict(np.array([img.shape[1]/2]).reshape(1, -1))
    
    #Dibujamos la línea obtenida por la regresión lineal y la línea original
    if(drawline):      
        #Agregamos el canal de color extra a la imagen en escala de grises para poder poner de color rojo la linea upper
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #Los puntos y0, y_mean y y1 son los puntos calculados por la regresion lineal
        pts = np.array([[x[0], y0], [x_mean, y_mean], [x[-1], y1]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        yo_mean = y[int(img.shape[1]/2)]
        pts_origin = np.array([[x[0], y[0]], [x_mean, yo_mean], [x[-1], y[-1]]], np.int32)
        pts_origin = pts.reshape((-1, 1, 2))               
        cv2.polylines(output, 
              [pts_origin], 
              isClosed = False,
              color = (255,0,0),
              thickness = 1)
        out = output.copy()
        cv2.polylines(out, 
              [pts], 
              isClosed = False,
              color = (0,0,255),
              thickness = 1)
        cv2.imshow('upper image', out)

    return y0, y1, int(y_mean)

def correct_slant(img, treshold=100):
    """Corrige slant del texto. Cursiva
    
    Parametros:
        img: Imagen en escala de grises. Positiva no normalizada: fondo valor 0 y negro valor 255 
        treshold: 
    """
    # Estimate slant angle
    angle = slant_angle(img, treshold_up = treshold, treshold_down = treshold)
    
    # convert image to to negative
    img = 255 - img
    
    # Add blanks in laterals to compensate the shear transformation cut 
    if angle>0:
        img = np.concatenate([img, np.zeros([img.shape[0], int(img.shape[0]*angle)])], axis=1)
    else:
        img = np.concatenate([np.zeros([img.shape[0], int(img.shape[0]*(-angle))]), img], axis=1)
        
    # Numero de columnas añadidas a la imagen
    # positions//2 permiten ajusta las posiciones de cada palabra si se tiene segmentandas antes de esta transformación
    positions = int(abs(img.shape[0]*angle))
        
    # shear matrix and affine transformation
    M = np.float32([[1, -angle, 0], [0, 1, 0]])
    img2 = cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    )
    
    
    return img2, angle, positions//2


def slant_angle(img, treshold_up=100, treshold_down=100):
    """ Calcular el angulo de inclinación
        - Check the upper neighboords of pixels with left blank
        - Utilizar despeus de hacer una mejora de contraste.
        - Usar despues de habr corregido el slope de la linea
    Parametros:
        img: Imagen en escala de grises. Positiva no normalizada: fondo valor 0 y negro valor 255 
        treshold_up: umbral de gris para decidir que algo es negro
        treshold_down: umbral de gris para decidir que algo es blanco
    """
    angle = []
    # Contadores para pixeles Centrales, Left y Right
    C = 0
    L = 0
    R = 0
    for w in range(0,img.shape[1]-1):
        for h in range(0,img.shape[0]-1):
            if img[h,w] > treshold_up and img[h, w-1] < treshold_down: # si pixel negro y blanco a la izquierda..
                if img[h-1, w-1] > treshold_up: # si arriba izquierda es negro
                    L +=1                    
                elif img[h-1, w] > treshold_up: # si arriba centro es negro
                    C += 1                    
                elif img[h-1, w+1] > treshold_up: # si arriba derecha es negro
                    R += 1
    logger.debug(f"Slant angle. Left, Center, Rigth, Angle: {L}, {C}, {R}, {(R-L)/(L+C+R)}")
    return np.arctan2((R-L),(L+C+R))


def normalizar(img, treshold=20):
    #Deteccion de la línea base
    y0_base, y1_base, y_mean_base, angle = detect_baseline(img, treshold=treshold)

    #Detección de la línea superior
    y0_upper, y1_upper, y_mean_upper = detect_upperlane(img, treshold=treshold)

    #Posiciones y de cada franja
    position_upper = max(0, int(min(y0_upper, y1_upper)))
    position_base = min(img.shape[0], int(max(y0_base, y1_base)))
    
    # Altura de cada franja
    h_upper = position_upper
    h_base = img.shape[0] - position_base
    h_core = position_base - position_upper
    
    #correccion de posiciones si core es pequeño
    if h_upper>5 & h_upper > h_core:
        position_upper -= 5
    if h_base > 5 & h_base > h_core:
        position_base += 5
    
    
    # Altura de cada franja
    h_upper = position_upper
    h_base = img.shape[0] - position_base
    h_core = position_base - position_upper
    logger.debug(position_upper, position_base, '-', h_upper, h_core, h_base)
    
    
    # Ajuste de la parte de descenders
    if h_base > h_core:
        r_baseline = max(h_core/h_base, 0.5)
        img_inf_rescaled = cv2.resize(img[position_base:,:], (img.shape[1], int(h_base * r_baseline)))
    elif h_base < 20:
        img_inf_rescaled = np.concatenate((img[position_base:,:], np.zeros((20-h_base, img.shape[1]))), axis=0)
    else:
        img_inf_rescaled = img[position_base:,:]
    
    # Ajuste de la parte de ascenders
    if h_upper > h_core:
        r_upperline = max(h_core/h_upper, 0.5)
        img_sup_rescaled = cv2.resize(img[:position_upper,:], (img.shape[1], int(h_upper * r_upperline)))
    elif h_upper < 20:
        img_sup_rescaled = np.concatenate((np.zeros((20-h_upper, img.shape[1])), img[:position_upper,:]), axis=0)
    else:
        img_sup_rescaled = img[:position_upper,:]
    
        
    logger.debug(f"Rescale areas: {h_upper}, {h_core}, {h_base} {img_inf_rescaled.shape} {img_sup_rescaled.shape}")
    
    
    img_rescaled = np.concatenate((img_sup_rescaled, img[position_upper:position_base,:], img_inf_rescaled), axis=0)

    return img_rescaled

def get_x_positions_line(pd_words, line_img, line, x_line, inc_positions):
    """ recupera las tupas de posiciones x de cada palabra de una linea de IAM
    y las corrige una posición x_line
    """
    x_positions = []
    words_list = []
    id_words_list = []
    for t in pd_words[(pd_words.line == line) & (pd_words.selected)].itertuples():
        x_positions += [(t.x - x_line + inc_positions, t.x - x_line + inc_positions + t.w)]
        words_list += [t.word]
        id_words_list += [t.id_word]
    logger.debug(f"x_positions: {x_positions}")
        
        
    x_positions_correct = []
    x_prev = 0
    for i in range(len(x_positions)-1):
        new_x = (x_positions[i][1] + x_positions[i+1][0]) // 2
        x_positions_correct += [(x_prev, new_x)]
        x_prev = new_x
    x_positions_correct += [(x_prev, x_positions[-1][1])] 
    
    
    x_positions_correct[-1] = (x_positions_correct[-1][0], line_img.shape[1])
    
    logger.debug(f"x_positions_correct: {x_positions_correct}")
    
    
    img_list = []
    for (x1,x2) in x_positions_correct:
        if x2-x1>=2:
            img_list += [line_img[:, x1:x2]]
        else:
            img_list += [line_img[:, x1:x1+2]]
            
    return img_list, id_words_list, words_list


# Normalize size of each word
def normalize_shape(img_list, x_size=192, y_size=48, plot=False):
    
    img_normalized_list = []

    for img in img_list:
        # ajuste de altura
        y, x = img.shape
        if y_size is not(None):
            img = cv2.resize(img, (max(2,int(x*(y_size/y))), y_size))

        # Recorte derecha e izquierda
        true_points = np.argwhere(img)
        if len(true_points)>0:
            # take the smallest points and use them as the top left of your crop
            top_left = true_points.min(axis=0)
            # take the largest points and use them as the bottom right of your crop
            bottom_right = true_points.max(axis=0)
            if bottom_right[1] - top_left[1] > 2:
                img = img[:, top_left[1]:bottom_right[1]+1]


        # Ajuste de anchura
        y, x = img.shape
        if x < x_size:
            img = np.concatenate([img, np.zeros([y, x_size - x])], axis=1)
        else:
            img = cv2.resize(img, (x_size, y_size))

        img_normalized_list += [img]
    
    if plot:
        fig = plt.figure()
        n = 1
        for img, img_norm in zip(img_list, img_normalized_list): 
            a = fig.add_subplot(len(img_list), 2, n)
            a.set_title('Original')
            fig.tight_layout()
            plt.imshow(255-img, cmap='gray')
            n += 1
            
            a = fig.add_subplot(len(img_list), 2, n)
            a.set_title('Normalized')
            fig.tight_layout()
            plt.imshow(255-img_norm, cmap='gray')
            n += 1
        
    return img_normalized_list


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def chop(image, offset,plot=False):
    dim = None
    (h, w) = image.shape[:2]

    images = []
    for i in range(0, w, offset):
       if i+28 < w:        
        chop = image[0:28,i:i+28].copy()
        if plot:        
            cv2.imshow('chop', chop) 
        images.append(chop)

    return images
     

# def get_wams_results(config, domain, lpm, trf, trl, tolerance, sigma, iota, kappa, msize):
    
#     # Round the values
#     max_value = trf.max()
#     min_value = trf.min()
   
#     n_labels = constants.n_labels
#     nmems = int(n_labels/lpm)

#     # Create the required associative memories.
#     ams = dict.fromkeys(range(nmems))
#     max_msize = 0 # para normalizacion
    
#     measures = np.zeros((constants.n_measures, nmems), dtype=np.float64)
#     entropy = np.zeros((nmems, ), dtype=np.float64)
#     behaviour = np.zeros((constants.n_behaviours, ))

#     for j in ams:
#         if msize > max_msize:
#             max_msize = msize
#         ams[j] = AssociativeMemory(domain, msize, tolerance, sigma, iota, kappa)

#     trf_rounded = np.round((trf-min_value) * (max_msize - 1) / (max_value-min_value)).astype(np.int16)

#     # Registration
#     for features, label in zip(trf_rounded, trl):
#         i = int(label/lpm)
#         ams[i].register(features)

#     # Calculate entropies
#     for j in ams:
#         entropy[j] = ams[j].entropy
    
#     #Aqui va la parte en donde se reconocerían las features de IAM DATASET 

#     return "!"


def increase_data():

    get_best_ams()



    return 'features'



def get_best_ams():
       
    if os.path.isfile(smac.statsfilename):
        df = pd.read_csv(smac.statsfilename, encoding='utf-8')
        #Get row whit the min F1 value  
        minValueIndex = df.idxmin()
        tolerance = df.iloc[minValueIndex[0], 9]
        sigma = df.iloc[minValueIndex[0], 10]
        iota = df.iloc[minValueIndex[0], 11]
        kappa = df.iloc[minValueIndex[0], 12]
        msize = df.iloc[minValueIndex[0], 13]

        prefix = constants.partial_prefix
        if prefix == constants.partial_prefix:
            suffix = constants.filling_suffix
        elif prefix == constants.full_prefix:
            suffix = constants.training_suffix

        #Aqui va un for sobre de todas las memorias y sus modelos
        i = 0
        training_features_filename = prefix + constants.features_name + suffix
        training_features_filename = constants.data_filename(training_features_filename, i)
        training_labels_filename = prefix + constants.labels_name + suffix
        training_labels_filename = constants.data_filename(training_labels_filename, i)

        training_iam_features = os.path.join('runs', 'features-iam-training-000.npy') 
        
        trf = np.load(training_features_filename)
        trl = np.load(training_labels_filename)

        triam = np.load(training_iam_features)

        min_value = trf.min()
        max_value = trf.max()

        min_value_iam = triam.min()
        max_value_iam = triam.max()

        nmems = constants.n_labels
        domain = constants.domain

        ams = dict.fromkeys(range(nmems))
        entropy = np.zeros((nmems, ), dtype=np.float64)
        max_msize = 0 # para normalizacion

        for j in ams:
            if msize > max_msize:
                max_msize = msize
            ams[j] = AssociativeMemory(domain, msize, tolerance, sigma, iota, kappa)
        
        trf_rounded = np.round((trf-min_value) * (max_msize - 1) / (max_value-min_value)).astype(np.int16)
        triam_rounded = np.round((triam-min_value_iam) * (max_msize - 1) / (max_value_iam-min_value_iam)).astype(np.int16)


        # Registration
        for features, label in zip(trf_rounded, trl):
            i = int(label)
            ams[i].register(features)

        # Calculate entropies
        for j in ams:
            entropy[j] = ams[j].entropy

        # Recognition
        response_size = 0
        
        for features in triam_rounded:            
            memories = []
            for k in ams:
                recognized, weight = ams[k].recognize(features)
                                
                print("Datos recognized",recognized)
                print("Datos weight",weight)
        # for features, label in zip(tef_rounded, tel):
        #     correct = int(label)

        #     memories = []
        #         for k in ams:
        #             recognized, weight = ams[k].recognize(features)

       
    else:
        return None


def proccess_line(pd_line, pd_words, iam_sources_path, destination_folder):
       
    # Read the entire line, an example of path is: databases\IAM\lines\b04\b04-00\b04-000-00.png
        line_filename = os.path.join(
            iam_sources_path,
            'lines',
            pd_line.id_line.split('-')[0],
            pd_line.id_line.split('-')[0]+"-"+pd_line.id_line.split('-')[1],
            pd_line.id_line+'.png'
        )

        image_contrast = enhance_contrast_otsu(line_filename)
        image_slope = slope_correction(255-image_contrast)
        line_img_no_slant, angle, inc_positions = correct_slant(255-image_slope)
        if abs(slant_angle(line_img_no_slant)) > 0.02:
                 line_img_no_slant, angle_2, inc_positions_2 = correct_slant(255-line_img_no_slant)
                 inc_positions = inc_positions + inc_positions_2
                 logger.debug(f"Slant segunda correccion: {angle_2} | {inc_positions_2}")

        img = np.float32(line_img_no_slant)
        img_rescaled = image_resize(img, height=28) 
        images = chop(img_rescaled, offset=7)

        return images

def preprocess_iam():

     #If destination folder does not exist, we created it.
    if not(os.path.exists(destination_folder)):
        os.makedirs(destination_folder) 

    #We check if we had previously created the file
    if not(os.path.exists(os.path.join(destination_folder,iam_filename))): 

        #Select the line to be process
        #In pd_lines are the lines of iam dataset with the following format:
        #columns=['id_line','segmentation_result','graylevel_binarize', 'num_components', 'x','y','w','h','word', 'id_writter', 'id_page']
        #In pd_words are the words of iam dataset with the following formart:
        #columns=['id_word','segmentation_result','graylevel_binarize','x','y','w','h','grammar_tag','word']
        images = []
        size = len(list(pd_lines[(pd_lines.id_line.isin(lines_selected))].itertuples()))
        loop = tqdm(total = size, position=0, leave=False)
        contador = 0
        for line in pd_lines[(pd_lines.id_line.isin(lines_selected))].itertuples(): #& (pd_lines.partition == 'trn')].itertuples(): Esta parte se utilizaria si se divide en entrenamiento el dataset
            try:
                imgs = proccess_line(line, pd_words, iam_sources_path, destination_folder)
                images.append(imgs)
                loop.set_description("Processing...".format(contador))
                loop.update(1)
                contador=contador+1
            except:
                logger.info(f"LINE ERROR: {line}")
                logger.info(f"Error: {sys.exc_info()[0]}")
        loop.close()
        
        np.savez(os.path.join(destination_folder,iam_filename), iam_filename=images)
        return os.path.join(destination_folder,iam_filename)
    else:
        return os.path.join(destination_folder,iam_filename)
        
                 
def main():

    file_path = preprocess_iam()
 
    # cv2.imshow('original image', cv2.imread(test_img_line, 0))
    # image_contrast = enhance_contrast_otsu(test_img_line)
    # cv2.imshow('contrast image', image_contrast)
    # image_slope = slope_correction(255-image_contrast)
    # cv2.imshow('slope image', image_slope)
    # line_img_no_slant, angle, inc_positions = correct_slant(255-image_slope)
    # cv2.imshow('slant image', line_img_no_slant)
    # if abs(slant_angle(line_img_no_slant)) > 0.02:
    #             line_img_no_slant, angle_2, inc_positions_2 = correct_slant(255-line_img_no_slant)
    #             inc_positions = inc_positions + inc_positions_2
    #             logger.debug(f"Slant segunda correccion: {angle_2} | {inc_positions_2}")
    #             cv2.imshow('slant image second correction', line_img_no_slant)
    
    # img = np.float32(line_img_no_slant) 

    # img_rescaled = image_resize(img, height=28)
    # cv2.imshow('imagen reescalada', img_rescaled)

    # chop(img_rescaled, offset=7)

    #img_rescaled = normalizar(img)
    #cv2.imshow('imagen reescalada', img_rescaled)
    
    #If destination folder does not exist, we created it.
    # if not(os.path.exists(destination_folder)):
    #     os.makedirs(destination_folder)  

    #Select the line to be process
    #In pd_lines are the lines of iam dataset with the following format:
    #columns=['id_line','segmentation_result','graylevel_binarize', 'num_components', 'x','y','w','h','word', 'id_writter', 'id_page']
    #In pd_words are the words of iam dataset with the following formart:
    #columns=['id_word','segmentation_result','graylevel_binarize','x','y','w','h','grammar_tag','word']
    # for line in pd_lines[(pd_lines.id_line.isin(lines_selected))].itertuples(): #& (pd_lines.partition == 'trn')].itertuples(): Esta parte se utilizaria si se divide en entrenamiento el dataset
    #     try:
    #         proccess_line(line, pd_words, iam_sources_path, destination_folder)
    #     except:
    #         logger.info(f"LINE ERROR: {line}")
    #         logger.info(f"Error: {sys.exc_info()[0]}")
       

    #img_words_list, id_words_list, words_list = get_x_positions_line(pd_words, img_rescaled, 'a02-000-00', 371, inc_positions)

    # Normalize shape and size of the words
    #plt.rcParams['figure.figsize'] = (20, 50)     
    #words_normalized_list = normalize_shape(img_words_list, x_size=192, y_size=28, plot=True)

    cv2.waitKey(0) 
    

if __name__ == "__main__":
    main()