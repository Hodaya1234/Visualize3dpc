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
Each chair's latent features are clustered. The points are colored in the matching cluster. 
<img src="https://user-images.githubusercontent.com/33811220/128633694-0f313b2f-cdd3-47e8-94d3-4acd2552984f.png" height="100">

### Clustering of the per point latent features over many examples of a single class
The fours airplanes' points' features are clustered together. We can see that same parts of different instances are form a single cluster.

1 
<img src="https://user-images.githubusercontent.com/33811220/128633034-62567a52-b5c3-432c-affe-cf6b4548b69e.gif" height="100">
2 
<img src="https://user-images.githubusercontent.com/33811220/128633040-19a97a1a-f749-47a7-ae34-490fe96fc068.gif" height="100">
3 
<img src="https://user-images.githubusercontent.com/33811220/128633042-f9b72082-603a-4915-9f84-f6d465a14ffb.gif" height="100">
4 
<img src="https://user-images.githubusercontent.com/33811220/128633043-3bc02519-02e6-4928-9915-679ad09b1fd2.gif" height="100">

### Class attentive response maps. (C-ARM)
In green we can wee all the points that were classified as the correct target class. In grey - all the other points.

Toilet:
<img src="https://user-images.githubusercontent.com/33811220/128632945-ad105411-4326-4d01-afad-cda933e15f99.gif" height="150">
Laptop:
<img src="https://user-images.githubusercontent.com/33811220/128632947-d85e23da-d13b-43b3-8414-06b68c98ef6a.gif" height="150">
Monitor:
<img src="https://user-images.githubusercontent.com/33811220/128632950-d2b8f638-5e35-4b16-bcf3-2683aababd9d.gif" height="150">


### Class activation maps. (CAM)
Airplane 23:
<img src="https://user-images.githubusercontent.com/33811220/128632524-8dba8361-f101-44d3-8fad-0c4ae9178f53.gif" height="100">
Airplane 28:
<img src="https://user-images.githubusercontent.com/33811220/128632526-cbd050f5-990a-4ac5-be17-f866acfa9a5f.gif" height="100">
Airplane 31:
<img src="https://user-images.githubusercontent.com/33811220/128632529-5d5f59df-5335-4c76-97ba-348b73685338.gif" height="100">


### Grad-CAM
Bottle 381:
<img src="https://user-images.githubusercontent.com/33811220/128632887-310e62fa-d1fb-4bea-bc90-3ad8e495ef62.gif" height="100">
Bottle 383:
<img src="https://user-images.githubusercontent.com/33811220/128632888-fe7adc80-ee38-491f-a045-baced4e2380e.gif" height="100">
Airplane 6:
<img src="https://user-images.githubusercontent.com/33811220/128632892-19813515-569a-4fe1-8f7a-20f6f2468312.gif" height="100">
Airplane 28:
<img src="https://user-images.githubusercontent.com/33811220/128632893-f2b7ae7d-7083-47b3-b2ba-516967e85ec3.gif" height="100">


### Point cloud synthesis of a single example

Lamp from a sphere:

<img src="https://user-images.githubusercontent.com/33811220/128633287-fe87b862-41c0-4234-b766-fda6010901f5.gif" height="200">

A table-Airplane hybrid by optimizing the input point cloud to have latent features that are close to a table and an airplane.
Optimization:

<img src="https://user-images.githubusercontent.com/33811220/128633383-85601f18-7a27-463a-af95-7c43f939f259.gif" height="200">

Final result: Airplane shape with four leg-like objects:

<img src="https://user-images.githubusercontent.com/33811220/128633487-8096bf33-cb13-4bad-845b-2662857cca9b.gif" height="200">





