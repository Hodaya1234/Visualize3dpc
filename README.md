# Visualization of the Point Cloud Transformer network

In this work we wanted to investigate and visualize the way that a transformer based 3d point net classification network works.
To do so we used the following methods:

- Clustering of the per point latent features of a single example.
- Clustering of the per point latent features over many examples of a single class, meaning clustering the union of all the points of several point clouds.
- Class attentive response maps. (C-ARM)
- Class activation maps. (CAM)
- Grad-CAM, meaning class activation response maps via the gradients of the loss w.r.t the latent features.
- Point cloud synthesis by optimizing a random input to have the latent features of a given input from the data.

## Results

### Clustering of the per point latent features of a single example.
![airplane 0](https://user-images.githubusercontent.com/33811220/128631419-2ba4b8d6-bb11-43da-9162-163b5a3b4e0d.gif)


### Clustering of the per point latent features over many examples of a single class

### Class attentive response maps. (C-ARM)

### Class activation maps. (CAM)

### Grad-CAM

### Point cloud synthesis