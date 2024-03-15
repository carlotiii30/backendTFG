# Protocolo de comunicación
Necesitamos un protocolo de comunicación entre el frontend y el backend.
En este caso buscamos uno que permita al frontend solicitar la generación de
imágenes a través del backend, que estará ejecutando modelos de GAN.

## Proceso
1. **Definición de comandos**: Definiremos un conjunto de comandos que el
frontend enviará al backend. Por ejemplo: `generate_image`, para solicitar la
generación de una nueva imagen.
2. **Estructura de los mensajes**: Utilizaremos un formato JSON para representar
los mensajes que se intercambiarán. Mandaremos un comando y unos parámetros.
3. **Procesamiento**: Cuando el backend recibe el comando se ejecutará la acción
requerida.
4. **Envío al frontend**: Una vez la acción se haya realizado, se enviará la
información requerida al frontend.