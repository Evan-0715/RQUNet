from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Conv2D,
    concatenate,
    Add,
    LeakyReLU,
    BatchNormalization)
from src.layer import Conv2D as RQConv2D
from src.layer import Conv2DTranspose as RQConv2DTranspose
from tensorflow import matmul, reshape, reduce_sum, transpose
from tensorflow.linalg import lu_matrix_inverse as matrix_inverse


def Projection(bridge, up, ssa, subspace_dim):
    b_, h_, w_, c_ = bridge.shape
    V = reshape(ssa, [-1, h_ * w_,
                      subspace_dim])
    V = V / (1e-6 + reduce_sum(abs(V), axis=1, keepdims=True))
    V_t = transpose(V, perm=[0, 2, 1])
    mat = matmul(V_t, V)
    try:
        mat_inv = matrix_inverse(mat)
        project_mat = matmul(mat_inv, V_t)
        bridge_ = reshape(bridge, [-1, h_ * w_, c_])
        project_feature = matmul(project_mat, bridge_)
        bridge = reshape(matmul(V, project_feature), [-1, h_, w_, c_])
        result = concatenate([bridge, up], axis=-1)
        return result
    except:
        return concatenate([bridge, up], axis=-1)


# RQUNet_RQDeconv_RQSSM_model
def RQUNet():
    # K=4 subspace-dim=16
    subspace_filters = 4
    input_image = layers.Input(shape=(512, 512, 3), dtype="float32")
    net = Conv2D(4, 3, padding="same")(input_image)  # 转换为四通道图片特征
    # downsample-1
    conv1 = RQConv2D(filters=8, kernel_size=3, padding="same")(net)
    conv1 = LeakyReLU()(conv1)
    conv1 = RQConv2D(filters=8, kernel_size=3, padding="same")(conv1)
    bridge1 = LeakyReLU()(conv1)
    pool1 = RQConv2D(filters=8, kernel_size=4, strides=2, padding="same")(bridge1)
    # downsample-2
    conv2 = RQConv2D(filters=16, kernel_size=3, padding="same")(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = RQConv2D(filters=16, kernel_size=3, padding="same")(conv2)
    bridge2 = LeakyReLU()(conv2)
    pool2 = RQConv2D(filters=16, kernel_size=4, strides=2, padding="same")(bridge2)

    # downsample-3
    conv3 = RQConv2D(filters=32, kernel_size=3, padding="same")(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = RQConv2D(filters=32, kernel_size=3, padding="same")(conv3)
    bridge3 = LeakyReLU()(conv3)
    pool3 = RQConv2D(filters=32, kernel_size=4, strides=2, padding="same")(bridge3)
    # downsample-4
    conv4 = RQConv2D(filters=64, kernel_size=3, padding="same")(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = RQConv2D(filters=64, kernel_size=3, padding="same")(conv4)
    bridge4 = LeakyReLU()(conv4)
    pool4 = RQConv2D(filters=64, kernel_size=4, strides=2, padding="same")(bridge4)

    conv5 = RQConv2D(filters=128, kernel_size=3, padding="same")(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = RQConv2D(filters=128, kernel_size=3, padding="same")(conv5)
    conv5 = LeakyReLU()(conv5)

    # upsample-4
    up4 = RQConv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    sub4 = concatenate([bridge4, up4], axis=-1)
    ssa4 = RQConv2D(filters=subspace_filters, kernel_size=3, padding="same")(sub4)  # 将来换成4通道
    up4 = Projection(bridge4, up4, ssa4, subspace_dim=subspace_filters * 4)
    conv6 = RQConv2D(filters=64, kernel_size=3, padding="same")(up4)
    conv6 = LeakyReLU()(conv6)
    conv6 = RQConv2D(filters=64, kernel_size=3, padding="same")(conv6)
    conv6 = LeakyReLU()(conv6)
    # upsample-3
    up3 = RQConv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6)
    sub3 = concatenate([bridge3, up3], axis=-1)
    ssa3 = RQConv2D(filters=subspace_filters, kernel_size=3, padding="same")(sub3)  # 将来换成4通道
    up3 = Projection(bridge3, up3, ssa3, subspace_dim=subspace_filters * 4)
    conv7 = RQConv2D(filters=32, kernel_size=3, padding="same")(up3)
    conv7 = LeakyReLU()(conv7)
    conv7 = RQConv2D(filters=32, kernel_size=3, padding="same")(conv7)
    conv7 = LeakyReLU()(conv7)
    # upsample-2
    up2 = RQConv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    sub2 = concatenate([bridge2, up2], axis=-1)
    ssa2 = RQConv2D(filters=subspace_filters, kernel_size=3, padding="same")(sub2)  # 将来换成4通道
    up2 = Projection(bridge2, up2, ssa2, subspace_dim=subspace_filters * 4)
    conv8 = RQConv2D(filters=16, kernel_size=3, padding="same")(up2)
    conv8 = LeakyReLU()(conv8)
    conv8 = RQConv2D(filters=16, kernel_size=3, padding="same")(conv8)
    conv8 = LeakyReLU()(conv8)
    # upsample-1
    up1 = RQConv2DTranspose(filters=8, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8)
    sub1 = concatenate([bridge1, up1], axis=-1)
    ssa1 = RQConv2D(filters=subspace_filters, kernel_size=3, padding="same")(sub1)  # 将来换成4通道
    up1 = Projection(bridge1, up1, ssa1, subspace_dim=subspace_filters * 4)
    conv9 = RQConv2D(filters=8, kernel_size=3, padding="same")(up1)
    conv9 = LeakyReLU()(conv9)
    conv9 = RQConv2D(filters=8, kernel_size=3, padding="same")(conv9)
    conv9 = LeakyReLU()(conv9)
    # recover 3-channel-R-G-B feature maps
    conv10 = Conv2D(3, 3, padding="same")(conv9)
    conv10 = Add()([input_image, conv10])
    model = models.Model(inputs=input_image, outputs=conv10)
    return model
