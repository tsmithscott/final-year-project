import tensorflow as tf


def load_trained_model():
    return tf.keras.models.load_model('model.h5')

# def plot_dataset_activation():
#     # Count the number of defective images
#     num_defective = sum([is_defective(img) for img in X])
    
#     # Determine the number of rows based on the number of defective images
#     num_rows = num_defective
    
#     # Create the subplots
#     fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8, 8))
    
#     # Initialize a counter for the row index
#     row_index = 0
    
#     # Iterate over the images
#     for i, img in enumerate(X):
#         if is_defective(img):
#             weights = model.layers[-1].get_weights()[0]
#             class_weights = weights[:, 0]

#             intermediate = tf.keras.Model(model.input, model.get_layer("block5_conv3").output)
#             conv_output = intermediate.predict(img[np.newaxis,:,:,:])
#             conv_output = np.squeeze(conv_output)

#             h = int(img.shape[0]/conv_output.shape[0])
#             w = int(img.shape[1]/conv_output.shape[1])

#             activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
#             out = np.dot(activation_maps.reshape((img.shape[0]*img.shape[1], 512)), class_weights).reshape(
#                 img.shape[0],img.shape[1])

#             axes[row_index][0].imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
#             axes[row_index][1].imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
#             axes[row_index][1].imshow(out, cmap='jet', alpha=0.35, extent=[0, img.shape[1], img.shape[0], 0])
#             axes[row_index][0].set_title('Defective')
#             axes[row_index][1].set_title('Activation Map')
            
#             # Increment the row index
#             row_index += 1

#     plt.tight_layout()
#     plt.show()
