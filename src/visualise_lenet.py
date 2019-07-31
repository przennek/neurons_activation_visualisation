def visualise(model, l_no, how_many_layers):
    layers = []
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        layers.append(layer)

    f, b = layers[l_no].get_weights()
    fmin, fmax = np.min(f), np.max(f)
    f = (f - fmin) / (fmax - fmin)
    j = 0
    for i in range(how_many_layers):
        fl = f[:,:,0, i]
        ax = plt.subplot(how_many_layers, 1, j+1)
        j += 1
        ax.set_yticks([])
        ax.set_xticks([])
        plt.imshow(fl, cmap='gray')
    plt.show()

def visualise_by_gd(model, img, layer_no, number_of_samples, boxing = 0):
    layers = []
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        layers.append(layer)

    img = np.expand_dims(img, axis=0)
    vmodel = Model(inputs=model.inputs, outputs=model.layers[layer_no].output)
    feature_maps = vmodel.predict(img)

    z = 0
    if boxing == 0:
        for _ in range(number_of_samples):
            ax = plt.subplot(number_of_samples, 1, z+1)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.imshow(feature_maps[0, :, :, z], cmap='gray')
            z += 1
    else:
        for _ in range(int(number_of_samples / boxing)):
            for _ in range(boxing):
               ax = plt.subplot(number_of_samples/boxing, boxing, z+1)
               ax.set_yticks([])
               ax.set_xticks([])
               plt.imshow(feature_maps[0, :, :, z], cmap='gray')
               z += 1
    plt.show()

