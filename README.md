# CNN para Blood-MNIST Dataset

Este proyecto implementa una Red Neuronal Convolucional (CNN) para clasificación de imágenes del dataset Blood-MNIST, organizada en archivos modulares para mejor mantenibilidad.

## Estructura del Proyecto

```
CNN_BloodMNIST/
├── CMakeLists.txt
├── README.md
├── main.cpp
├── Enums.h
├── Image.h
├── Image.cpp
├── ConvolutionKernel.h
├── ConvolutionKernel.cpp
├── ActivationFunction.h
├── ActivationFunction.cpp
├── PoolingLayer.h
├── PoolingLayer.cpp
├── ConvolutionLayer.h
├── ConvolutionLayer.cpp
├── BloodMNISTLoader.h
├── BloodMNISTLoader.cpp
├── ConvolutionalNeuralNetwork.h
└── ConvolutionalNeuralNetwork.cpp
```

## Componentes del Sistema

### Clases Principales

1. **Image**: Estructura para representar imágenes con sus datos y etiquetas
2. **ConvolutionKernel**: Maneja los pesos y bias de los kernels de convolución
3. **ActivationFunction**: Implementa funciones de activación (ReLU, Sigmoid, Tanh, Leaky ReLU)
4. **PoolingLayer**: Implementa operaciones de pooling (Max, Min, Average)
5. **ConvolutionLayer**: Capa de convolución completa con activación
6. **BloodMNISTLoader**: Cargador de datos desde archivos CSV
7. **ConvolutionalNeuralNetwork**: Clase principal que combina todas las capas

### Enumeraciones

- **PoolingType**: Tipos de pooling disponibles
- **ActivationType**: Funciones de activación disponibles

## Compilación en CLion

### Paso 1: Crear el Proyecto
1. Abre CLion
2. Selecciona "New CMake Project" o "Open" si ya tienes una carpeta
3. Copia todos los archivos .h, .cpp y CMakeLists.txt en tu directorio de proyecto

### Paso 2: Configurar CMake
1. CLion debería detectar automáticamente el archivo CMakeLists.txt
2. Si no lo hace, ve a `File` → `Reload CMake Project`
3. Asegúrate de que el compilador esté configurado correctamente

### Paso 3: Compilar y Ejecutar
1. Selecciona el target `CNN_BloodMNIST` en la configuración de Run
2. Presiona `Ctrl+F9` para compilar o `Shift+F10` para compilar y ejecutar
3. También puedes usar los botones de Build/Run en la interfaz

## Uso del Sistema

### Carga de Datos
```cpp
BloodMNISTLoader loader;
bool success = loader.loadFromCSV("blood_dataset.csv");
```

### Configuración de la CNN
```cpp
ConvolutionalNeuralNetwork cnn;
cnn.addConvolutionLayer(3, 32, 3, ActivationType::RELU);
cnn.addPoolingLayer(2, PoolingType::MAX);
cnn.addConvolutionLayer(32, 64, 3, ActivationType::RELU);
cnn.addPoolingLayer(2, PoolingType::MAX);
```

### Procesamiento de Imágenes
```cpp
auto output = cnn.forward(input_image);
```

## Características Implementadas

- ✅ Convolución 2D con padding y stride configurables
- ✅ Múltiples funciones de activación
- ✅ Pooling (Max, Min, Average)
- ✅ Carga de datos desde CSV
- ✅ Arquitectura modular y extensible
- ✅ Validación de dimensiones
- ✅ Inicialización aleatoria de pesos

## Arquitectura por Defecto

La CNN implementa la siguiente arquitectura:
1. **Conv1**: 3→32 canales, kernel 3×3, ReLU
2. **MaxPool**: 2×2
3. **Conv2**: 32→64 canales, kernel 3×3, ReLU  
4. **MaxPool**: 2×2
5. **Conv3**: 64→128 canales, kernel 3×3, ReLU
6. **AvgPool**: 2×2

## Dataset Blood-MNIST

El sistema está diseñado para trabajar con el dataset Blood-MNIST que contiene:
- **8 clases**: Basophil, Eosinophil, Erythroblast, Immature granulocytes, Lymphocyte, Monocyte, Neutrophil, Platelet
- **Dimensiones**: 28×28×3 (RGB)
- **Formato**: CSV con etiquetas y píxeles normalizados

## Extensiones Futuras

- [ ] Backpropagation y entrenamiento
- [ ] Batch normalization
- [ ] Dropout layers
- [ ] Diferentes optimizadores
- [ ] Métricas de evaluación
- [ ] Serialización de modelos

## Compilación Manual (sin CLion)

Si prefieres compilar manualmente:

```bash
mkdir build
cd build
cmake ..
make
./CNN_BloodMNIST
```

## Requisitos

- C++17 o superior
- CMake 3.16 o superior
- Compilador compatible (GCC, Clang, MSVC)
