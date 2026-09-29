import tensorflow as tf

print("TF:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs encontradas:", gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU configurada com sucesso!")
else:
    print("Nenhuma GPU detectada.")
