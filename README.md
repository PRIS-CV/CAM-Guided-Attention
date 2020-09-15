# CAM-based-spatial-attention-for-FGVC
## Good Luck!
***Result on Birds, Cars, with VGG16 and ResNet18, with pre-training***


|Method                      | Backbone       | Birds Acc@1 | Air Acc@1     | Cars Acc@1     | alpha           | 
|:----:                      | :-:            | :----:      | :----:        | :----:         | :----:          | 
|baseline                    | VGG16/ResNet50 |  84.75/86.5 | 88.4 /91.67   | 89.6 / 92.4    |    N/A          |
|channel attention           | VGG16/ResNet50 |  84.8 /86.78| 90.12/91.87   | 89.75/ 93.1    |    N/A          |
|spatial attention           | VGG16/ResNet50 |  84.92/86.99| 90.35/91.91   | 91.12/93.35    |      N/A        |
|mixed attention             | VGG16/ResNet50 |  85.29/87.14| 91.04/92.1    | 91.89 /93.34   |     N/A         | 
|CAM based spatial attention | VGG16/ResNet50 |   /         |    /92.52     | 92.61 / 93.61  | /,1/2,3         | 
