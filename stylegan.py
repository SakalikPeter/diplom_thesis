from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU, Concatenate, Embedding
from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import pandas as pd
import wandb
from utils import *
from layers import *
from losses import *

args = vars(parse_arguments())
DATA_PATH = args['data_path']
WANDB_KEY = args['wandb_key']
IMAGE_RESOLUTION = args['resolution']
PENALTY_CONST = args['wgan_penalty_const']
DISC_ITER = args['discriminator_iterations']
ITERATIONS = args['iterations']
INTERVAL = args['interval']
MAX_LOG2RES = args['max_log2res']
GROW_MODEL = args['grow_model']

BATCH_SIZE = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 4, 8: 4, 9: 4, 10: 4}
TRAIN_STEP_RATIO = {2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 1, 8: 1, 9: 2, 10: 4}
FILTER_SIZE = {0: 512, 1: 512, 2: 512, 3: 512, 4: 512, 5: 256, 6: 256, 7: 128, 8: 128, 9: 16, 10: 16}
OP = {'learning_rate': 5e-4, 'beta_1': 0., 'beta_2': 0.99, 'epsilon': 1e-8}
ORGAN_OUTPUT = {'[0]': [1.0, 0.0, 0.0, 0.0],
                '[1]': [0.0, 1.0, 0.0, 0.0],
                '[2]': [0.0, 0.0, 1.0, 0.0],
                '[3]': [0.0, 0.0, 0.0, 1.0]}
CELLS_OUTPUT = {'[0 0 0 0]': [1.0, 0.0, 0.0, 0.0],
                '[1 1 1 1]': [0.0, 1.0, 0.0, 0.0],
                '[2 2 2 2]': [0.0, 0.0, 1.0, 0.0],
                '[3 3 3 3]': [0.0, 0.0, 0.0, 1.0],
                '[0 0 1 1]': [1.0, 1.0, 0.0, 0.0],
                '[0 0 2 2]': [1.0, 0.0, 1.0, 0.0],
                '[0 0 3 3]': [1.0, 0.0, 0.0, 1.0],
                '[1 1 2 2]': [0.0, 1.0, 1.0, 0.0],
                '[1 1 3 3]': [0.0, 1.0, 0.0, 1.0],
                '[2 2 3 3]': [0.0, 0.0, 1.0, 1.0],
                '[0 0 1 2]': [1.0, 1.0, 1.0, 0.0],
                '[0 1 1 3]': [1.0, 1.0, 0.0, 1.0],
                '[0 2 2 3]': [1.0, 0.0, 1.0, 1.0],
                '[1 2 3 3]': [0.0, 1.0, 1.0, 1.0],
                '[0 1 2 3]': [1.0, 1.0, 1.0, 1.0]}
INTERPOLATION = 'nearest'


class StyleGAN:
    def __init__(self, z_dim=512, resolution=512, start_log2_res=2):
        self.start_log2_res = start_log2_res
        self.resolution = resolution
        self.log2_resolution = int(np.log2(resolution))
        self.log2_res_to_filter_size = FILTER_SIZE
        self.z_dim = z_dim
        self.opt_init = OP
        self.optimizer_discriminator = Adam(**self.opt_init)
        self.g_loss = None
        self.d_loss = None
        self.alpha = np.array([[1]], dtype=np.float32)
        self.df = pd.read_csv(f'{DATA_PATH}/v4/names.csv', index_col=0)
        self.n_rows = len(self.df.index)

        self.to_rgb = {}
        self.generator_blocks = {}
        self.noise_inputs = {}
        self.w_inputs = {}
        self.from_rgb = {}
        self.discriminator_blocks = {}

        self.mapping = None
        self.generator = None
        self.discriminator = None
        self.model = None

    def init_models(self):
        # ---------------------------
        # initialize generator
        # ---------------------------
        dummy_alpha = Input(shape=1, name='DummyAlpha')
        input_const = Input(shape=(4, 4, 512), name='ConstInput')
        w = Input(shape=(8, 512))

        output = self.generator_blocks[2]([input_const, w[:, 0], self.noise_inputs[2]])
        rgb = self.to_rgb[2](output)
        self.generator = Model([input_const, w, dummy_alpha, self.noise_inputs], rgb)

        # ---------------------------
        # initialize discriminator
        # ---------------------------
        input_image = Input(shape=(4, 4, 6))
        alpha = Input(shape=1)
        x = self.from_rgb[2](input_image)
        y1, y2, y3 = self.discriminator_blocks[2](x)
        self.discriminator = Model([input_image, alpha], [y1, y2, y3], name='discriminator_4x4')
        self.discriminator.trainable = False

        # ---------------------------
        # initialize model
        # ---------------------------
        y1, y2, y3 = self.discriminator([self.generator.output, self.generator.input[2]])
        self.model = Model([self.generator.input], [y1, y2, y3])
        self.model.compile(loss=[wasserstein_loss, 'binary_crossentropy', 'categorical_crossentropy'],
                           optimizer=Adam(**self.opt_init))

    # ---------------------------
    # build all
    # ---------------------------
    def build_mapping(self):
        z = Input(shape=self.z_dim)
        w = PixelNorm()(z)

        cls = Input(shape=4)
        c = Embedding(4, 512)(cls)
        c = Flatten()(c)
        c = Dense(512)(c)
        c = PixelNorm()(c)

        org = Input(shape=1)
        o = Embedding(4, 512)(org)
        o = Flatten()(o)
        o = PixelNorm()(o)

        w = Concatenate()([w, c, o])
        for i in range(8):
            if i % 2 == 1:
                w = Concatenate()([w, c, o])
            w = Dense(512, lrmul=0.01)(w)
            w = LeakyReLU(0.2)(w)

        w = tf.tile(tf.expand_dims(w, 1), (1, 8, 1))
        self.mapping = Model([z, cls, org], w, name='mapping')

    def build_all_generators(self):

        for log2_res in range(2, self.log2_resolution + 1):
            res = 2 ** log2_res
            filter_n = self.log2_res_to_filter_size[log2_res]

            self.noise_inputs[log2_res] = Input(shape=(res, res, 1), name=f'noise_{res}x{res}')
            self.w_inputs[log2_res] = Input(shape=512)

            self.to_rgb[log2_res] = self.build_to_rgb(res, filter_n)

            input_shape = (4, 4, 512) if log2_res == 2 else self.generator_blocks[log2_res - 1].output[0].shape
            gen_block = self.build_generator_block(log2_res, input_shape)
            self.generator_blocks[log2_res] = gen_block

    def build_all_discriminators(self):

        for log2_res in range(self.log2_resolution, 1, -1):
            res = 2 ** log2_res
            filter_n = self.log2_res_to_filter_size[log2_res]
            input_shape = (res, res, filter_n)

            self.from_rgb[log2_res] = self.build_from_rgb(res, filter_n)
            self.discriminator_blocks[log2_res] = self.build_discriminator_block(log2_res, input_shape)

        # last block at 4x4 resolution
        log2_res = 2
        filter_n = self.log2_res_to_filter_size[log2_res]
        res = 2 ** log2_res
        input_shape = (res, res, filter_n)

        self.from_rgb[log2_res] = self.build_from_rgb(4, filter_n)
        self.discriminator_blocks[log2_res] = self.build_discriminator_base(input_shape)

    # ---------------------------
    # building blocks
    # ---------------------------
    def build_generator_block(self, log2_res, input_shape):
        res = int(2 ** log2_res)
        res_name = f'{res}x{res}'
        filter_n = self.log2_res_to_filter_size[log2_res]

        w = Input(shape=512)
        noise = Input(shape=(res, res, 1))
        input_tensor = Input(shape=input_shape)
        x = input_tensor

        if log2_res > 2:
            x = UpSampling2D((2, 2), interpolation=INTERPOLATION)(x)
            x = Conv2D(filter_n, 3, name=f'gen_{res_name}_conv1')(x)

        x = AddNoise()([x, noise])
        x = LeakyReLU(0.2)(x)
        x = InstanceNormalization()(x)
        x = AdaIN()([x, w])

        # ADD NOISE
        x = Conv2D(filter_n, 3, name=f'gen_{res_name}_conv2')(x)
        x = AddNoise()([x, noise])
        x = LeakyReLU(0.2)(x)
        x = InstanceNormalization()(x)
        x = AdaIN()([x, w])

        return Model([input_tensor, w, noise], x, name=f'genblock_{res}_x_{res}')

    def build_discriminator_block(self, log2_res, input_shape):
        filter_n = self.log2_res_to_filter_size[log2_res]
        res = 2 ** log2_res
        input_tensor = Input(shape=input_shape)

        # First conv
        x = Conv2D(filter_n, 3)(input_tensor)
        x = LeakyReLU(0.2)(x)

        # Second conv + downsample
        filter_n = self.log2_res_to_filter_size[log2_res - 1]
        x = Conv2D(filter_n, 3)(x)
        x = LeakyReLU(0.2)(x)
        x = AveragePooling2D((2, 2))(x)

        return Model(input_tensor, x, name=f'disc_block_{res}_x_{res}')

    def build_discriminator_base(self, input_shape):
        input_tensor = Input(shape=input_shape)

        x = MinibatchStd()(input_tensor)

        x = Conv2D(512, 3, name='gen_4x4_conv1')(x)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)

        x = Dense(512, name='gen_4x4_dense1')(x)
        x = LeakyReLU(0.2)(x)

        y1 = Dense(1, name='gen_4x4_dense2')(x)
        y2 = Dense(4, name='gen_4x4_dense3')(x)
        y3 = Dense(4, name='gen_4x4_dense4')(x)

        return Model(input_tensor, [y1, y2, y3], name='discriminator_base')

    def build_to_rgb(self, res, filter_n):
        return Sequential([Input(shape=(res, res, filter_n)),
                           Conv2D(6, 1, gain=1, activation=None, name=f'to_rgb_{res}x{res}_conv')],
                          name=f'to_rgb_{res}x{res}')

    def build_from_rgb(self, res, filter_n):
        return Sequential([Input(shape=(res, res, 6)),
                           Conv2D(filter_n, 1, name=f'from_rgb_{res}x{res}_conv'),
                           LeakyReLU(0.2)],
                          name=f'from_rgb_{res}x{res}')

    # ---------------------------
    # grow model
    # ---------------------------
    def grow_discriminator(self, log2_res):
        res = 2 ** log2_res

        input_image = Input(shape=(res, res, 6))
        alpha = Input(shape=(1))

        x = self.from_rgb[log2_res](input_image)
        x = self.discriminator_blocks[log2_res](x)

        downsized_image = AveragePooling2D((2, 2))(input_image)
        y = self.from_rgb[log2_res - 1](downsized_image)

        x = FadeIn()(alpha, x, y)
        for i in range(log2_res - 1, 1, -1):
            x = self.discriminator_blocks[i](x)

        self.discriminator = Model([input_image, alpha], x, name=f'discriminator_{res}_x_{res}')
        self.optimizer_discriminator = Adam(**self.opt_init)

    def grow_generator(self, log2_res):
        res = 2 ** log2_res

        alpha = Input(shape=1)
        w = Input(shape=(8, 512))
        input_const = Input(shape=(4, 4, 512))

        x = self.generator_blocks[2]([input_const, w[:, 0], self.noise_inputs[2]])

        for i in range(3, log2_res):
            x = self.generator_blocks[i]([x, w[:, i - 2], self.noise_inputs[i]])

        old_rgb = self.to_rgb[log2_res - 1](x)
        old_rgb = UpSampling2D((2, 2), interpolation=INTERPOLATION)(old_rgb)

        x = self.generator_blocks[log2_res]([x, w[:, log2_res - 2], self.noise_inputs[log2_res]])

        new_rgb = self.to_rgb[log2_res](x)
        rgb = FadeIn()(alpha, new_rgb, old_rgb)

        self.generator = Model([input_const, w, alpha, self.noise_inputs], rgb, name=f'generator_{res}_x_{res}')

    def grow_model(self, log2_res):
        self.grow_generator(log2_res)
        self.grow_discriminator(log2_res)

        self.discriminator.trainable = False
        y1, y2, y3 = self.discriminator([self.generator.output, self.generator.input[2]])
        self.model = Model(self.generator.input, [y1, y2, y3])
        self.model.compile(loss=[wasserstein_loss, 'binary_crossentropy', 'categorical_crossentropy'],
                           optimizer=Adam(**self.opt_init))

    # ---------------------------
    # others
    # ---------------------------
    def generate(self, z, log2_res):
        batch_size = z.shape[0]
        const_input = tf.ones((batch_size, 4, 4, 512))

        noise = generate_noise(batch_size, self.log2_resolution)
        _, cells, organ = load_real_samples(self.df, batch_size, DATA_PATH, log2_res, self.n_rows)

        w = self.mapping([z, cells, organ])
        images = self.generator([const_input, w, self.alpha, noise])
        images = np.clip((images * 0.5 + 0.5) * 255, 0, 255)

        return images.astype(np.uint8)

    def checkpoint(self, log2_res, step, state):
        res = 2 ** log2_res
        prefix = f'res_{res}x{res}_{step}_{state}'
        images = []

        for _ in range(4):
            z = tf.random.normal((4, self.z_dim))
            imgs = self.generate(z, log2_res)
            images.append(imgs)

        images = np.concatenate(images, axis=0)

        plot_images(images[:, :, :, 0:3], log2_res, f"./save_images/images{prefix}.jpg")
        plot_images(images[:, :, :, 3:], log2_res, f"./save_images/masks{prefix}.jpg")

    def load_checkpoint(self):
        self.mapping = tf.keras.models.load_model(f'{DATA_PATH}/saved_models/mapping')

        for i in range(2, MAX_LOG2RES+1):
            print(i)
            self.discriminator_blocks[i]= tf.keras.models.load_model(f'{DATA_PATH}/saved_models/d_{i}')
            self.generator_blocks[i]= tf.keras.models.load_model(f'{DATA_PATH}/saved_models/g_{i}')

            self.to_rgb[i]= tf.keras.models.load_model(f'{DATA_PATH}/saved_models/to_rgb_{i}')
            self.from_rgb[i]= tf.keras.models.load_model(f'{DATA_PATH}/saved_models/from_rgb_{i}')

        self.grow_model(MAX_LOG2RES+GROW_MODEL)
        self.start_log2_res = MAX_LOG2RES+GROW_MODEL
        print('loaded succesfully!!!')

    # ---------------------------
    # train
    # ---------------------------
    def train_discriminator_wgan_gp(self, real_images, real_cells, real_organ):
        cells_true = np.array([CELLS_OUTPUT[str(x[0])] for x in real_cells])
        organ_true = np.array([ORGAN_OUTPUT[str(x[0])] for x in real_organ])

        batch_size = real_images.shape[0]
        const_input = tf.ones((batch_size, 4, 4, 512))
        noise = generate_noise(batch_size, self.log2_resolution)

        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        z = tf.random.normal((batch_size, self.z_dim))
        w = self.mapping([z, real_cells, real_organ])
        fake_images = self.generator([const_input, w, self.alpha, noise])

        with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:
            epsilon = tf.random.uniform((batch_size, 1, 1, 1))
            interpolates = epsilon * real_images + (1 - epsilon) * fake_images
            gradient_tape.watch(interpolates)

            # forward pass
            pred_fake, pred_class_fake, pred_organ_fake = self.discriminator([fake_images, self.alpha])
            pred_real, pred_class_real, pred_organ_real = self.discriminator([real_images, self.alpha])
            pred_fake_grad, _, _ = self.discriminator([interpolates, self.alpha])

            # calculate losses
            loss_fake = wasserstein_loss(fake_labels, pred_fake)
            loss_real = wasserstein_loss(real_labels, pred_real)
            loss_fake_grad = wasserstein_loss(fake_labels, pred_fake_grad)
            bce_loss = binary_cross_entropy((cells_true, pred_class_fake), (cells_true, pred_class_real))
            cce_loss = categorical_cross_entropy((organ_true, pred_organ_fake), (organ_true, pred_organ_real))

            # gradient penalty
            gradients_fake = gradient_tape.gradient(loss_fake_grad, [interpolates])
            gradient_penalty = gradient_loss(gradients_fake, PENALTY_CONST)

            # drift loss
            all_pred = tf.concat([pred_fake, pred_real], axis=0)
            drift_loss = 0.001 * tf.reduce_mean(all_pred ** 2)

            total_loss = loss_fake + loss_real + gradient_penalty + drift_loss

        # apply gradients
        gradients = total_tape.gradient([total_loss, bce_loss, cce_loss], self.discriminator.variables)
        self.optimizer_discriminator.apply_gradients(zip(gradients, self.discriminator.variables))

        return [total_loss, bce_loss, cce_loss]

    def train_step(self, log2_res, state):
        real_images = None
        real_cells = None
        real_organ = None

        for _ in range(DISC_ITER):
            real_images, real_cells, real_organ = load_real_samples(self.df, BATCH_SIZE[log2_res], DATA_PATH, log2_res,
                                                                     self.n_rows)
            self.d_loss = self.train_discriminator_wgan_gp(real_images, real_cells, real_organ)

        batch_size = real_images.shape[0]
        real_labels = tf.ones(batch_size)
        cells_true = np.array([CELLS_OUTPUT[str(x[0])] for x in real_cells])
        organ_true = np.array([ORGAN_OUTPUT[str(x[0])] for x in real_organ])

        z = tf.random.normal((batch_size, self.z_dim))
        w = self.mapping([z, real_cells, real_organ])

        const_input = tf.ones((batch_size, 4, 4, 512))
        noise = generate_noise(batch_size, self.log2_resolution)

        self.g_loss = self.model.train_on_batch([const_input, w, self.alpha, noise],
                                                [real_labels, cells_true, organ_true])

        wandb.log({
            f"Generátor Wasserstein {2 ** log2_res} {state}": self.g_loss[1],
            f"Generátor BCE {2 ** log2_res} {state}": self.g_loss[2],
            f"Generátor CCE {2 ** log2_res} {state}": self.g_loss[3],
            f"Kritik Wasserstein {2 ** log2_res} {state}": self.d_loss[0],
            f"Kritik BCE {2 ** log2_res} {state}": self.d_loss[1],
            f"Kritik CCE {2 ** log2_res} {state}": self.d_loss[2],
        })

    def train(self, default_steps=1000, interval=500):

        for log2_res in range(self.start_log2_res, self.log2_resolution + 1, 1):
            states = ['TRANSITION', 'STABLE']
            states = ['STABLE']

            for state in states:
                if state == 'TRANSITION' and log2_res == 2:
                    continue

                steps = int(TRAIN_STEP_RATIO[log2_res] * default_steps)
                interval = int(TRAIN_STEP_RATIO[log2_res] * interval)

                for step in range(steps):
                    self.alpha = np.array([[step / steps]]) if state == 'TRANSITION' else 1.

                    if step % interval == 0:
                        self.checkpoint(log2_res, step, state)

                    self.train_step(log2_res, state)

                    if state == 'STABLE' and step % 2000 == 0:
                        self.mapping.save(f'saved_models/256_{step}_step/mapping')
                        for i in range(2, log2_res + 1):
                            self.to_rgb[i].save(f'saved_models/256_{step}_step/to_rgb_{i}')
                            self.from_rgb[i].save(f'saved_models/256_{step}_step/from_rgb_{i}')

                            self.discriminator_blocks[i].save(f'saved_models/256_{step}_step/d_{i}')
                            self.generator_blocks[i].save(f'saved_models/256_{step}_step/g_{i}')
                    
                        wandb.save(f'saved_models/256_{step}_step/*/*.pb')
                        wandb.save(f'saved_models/256_{step}_step/*/variables/variables.index')
                        wandb.save(f'saved_models/256_{step}_step/*/variables/variables.data*')

                self.checkpoint(log2_res, steps, state, )

            if log2_res != self.log2_resolution:
                self.grow_model(log2_res + 1)

            self.mapping.save(f'saved_models/{log2_res}/mapping')
            for i in range(2, log2_res+1):
                self.to_rgb[i].save(f'saved_models/{log2_res}/to_rgb_{i}')
                self.from_rgb[i].save(f'saved_models/{log2_res}/from_rgb_{i}')
    
                self.discriminator_blocks[i].save(f'saved_models/{log2_res}/d_{i}')
                self.generator_blocks[i].save(f'saved_models/{log2_res}/g_{i}')
            
            wandb.save(f'saved_models/{log2_res}/*/*.pb')
            wandb.save(f'saved_models/{log2_res}/*/variables/variables.index')
            wandb.save(f'saved_models/{log2_res}/*/variables/variables.data*')


gan = StyleGAN(resolution=IMAGE_RESOLUTION)

gan.build_mapping()
gan.build_all_generators()
gan.build_all_discriminators()
gan.init_models()

wandb.login(key=WANDB_KEY)
wandb.init(project='DP', entity='pepe_raider')

if MAX_LOG2RES + GROW_MODEL:
    gan.load_checkpoint()

gan.train(ITERATIONS, INTERVAL)
