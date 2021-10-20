import numpy as np
import torch
import torch.nn as nn
import torch.distributions as pdfun
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from data_prepare.model import MIWAE
from data_prepare.dataframe import dataframe

"""
Find a data from here
https://archive.ics.uci.edu/ml/datasets.php
"""

class Trainer:
    def __init__(self, model, n_samples):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eps = torch.finfo(float).eps #a small epsilon value
        self.n_samples = n_samples
        self.model = model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, momentum=0.9)


        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=2
        )

    def ELBO(self, outdic):
        # the MIWAE ELBO 
        lpxz, lqzx, lpz = outdic['lpxz'], outdic['lqzx'], outdic['lpz'] 
        l_w = lpxz + lpz - lqzx # importance weights in the paper eq(4) 
        log_sum_w = torch.logsumexp(l_w, dim=1) #dim=1: samples
        log_avg_weight = log_sum_w - torch.log(x = self.n_samples.type(torch.FloatTensor))
        #should be $l(\theda)$ in the paper
        # TODO: check self.n_samples should be one of output dimensions
        # .shape[]
        return torch.reduce_mean(log_avg_weight, dim=-1)

    def gauss_loss(x, s, mu, log_sig2):
        #p(x | z) with Gauss z
        p_x_given_z = - (np.log(2 * np.pi) + log_sig2 + torch.square(x - mu) / (torch.exp(log_sig2) + self.eps))/2.
        return torch.reduce_sum(p_x_given_z * s, dim=-1)  # sum over d-dimension

    def bernoulli_loss(x, s, y):
        #p(x | z) with bernoulli z
        p_x_given_z = x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        return torchtf.reduce_sum(s * p_x_given_z, dim=-1)  # sum over d-dimension
        
    def KL_loss(q_mu, q_log_sig2):
        KL = 1 + q_log_sig2 - torch.square(q_mu) - torch.exp(q_log_sig2)
        return - torch.reduce_sum(KL, dim=1)/2.




    def local_train(self):
        """A regular training procedure.
        :return:
        """
        self.model.train()
        for epoch in range(2):
            # set the model to train mode
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.
        Args:
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.
        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """
        # retrieve model weights download from server's shareable
        model_weights = shareable[ShareableKey.MODEL_WEIGHTS]

        # update local model weights with received weights
        self.model.load_state_dict(
            {k: torch.as_tensor(v) for k, v in model_weights.items()}
        )

        self.local_train()

        # build the shareable
        shareable = Shareable()
        shareable[ShareableKey.META] = {FLConstants.NUM_STEPS_CURRENT_ROUND: 1}
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHTS
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = {
            k: v.cpu().numpy() for k, v in self.model.state_dict().items()
        }

        self.logger.info("Local epochs finished.  Returning shareable")
        return shareable





model = MIWAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
def train(self, epoch, device, optimizer = optim.Adam, lr=1e-3):
    model = self.to(device)
    self.optimizer = optimizer(self.parameters(), lr=1e-3)




        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, q_z = model(data)
            loss = loss_function(recon_batch, data, q_z)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))




 
        # ---- MIWAE:
        self.MIWAE = self.ELBO(self.log_p_x_given_z, self.log_q_z_given_x, self.log_p_z)

        # ---- loss
        self.loss = -self.MIWAE


        # ---- training stuff
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        self.optimizer = tf.train.AdamOptimizer()
        if self.testing:
            tvars = tf.trainable_variables(scope='encoder')
        else:
            tvars = tf.trainable_variables()
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step, var_list=tvars)

        self.sess.run(tf.global_variables_initializer())

        if permutation_invariance:
            svars = tf.trainable_variables('decoder')
            svars.append(self.global_step)
            self.saver = tf.train.Saver(svars)
        else:
            self.saver = tf.train.Saver()

        tf.summary.scalar('Evaluation/loss', self.loss)
        tf.summary.scalar('Evaluation/pxz', tf.reduce_mean(self.log_p_x_given_z))
        tf.summary.scalar('Evaluation/qzx', tf.reduce_mean(self.log_q_z_given_x))
        tf.summary.scalar('Evaluation/pz', tf.reduce_mean(self.log_p_z))

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.FileWriter(name + '/tensorboard/miwae_train/{}/'.format(timestamp),
                                                  self.sess.graph)
        self.val_writer = tf.summary.FileWriter(name + '/tensorboard/miwae_val/{}/'.format(timestamp),
                                                self.sess.graph)
        self.summaries = tf.summary.merge_all()

    def train_batch(self, batch_size):

        x_batch = self.X[self.batch_pointer: self.batch_pointer + batch_size, :]
        s_batch = self.S[self.batch_pointer: self.batch_pointer + batch_size, :]

        _, _loss, _step = \
            self.sess.run([self.train_op, self.loss, self.global_step],
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

        self.tick_batch_pointer(batch_size)

        return _loss

    def val_batch(self):

        batch_size = 100
        val_loss = 0.0
        pxz = 0.0
        pz = 0.0
        qzx = 0.0
        n_val_batches = len(self.Xval) // batch_size

        for i in range(n_val_batches):

            x_batch = self.Xval[i * batch_size: (i + 1) * batch_size]
            s_batch = self.Sval[i * batch_size: (i + 1) * batch_size]

            _loss, _pxz, _qzx, _pz, _step = \
                self.sess.run([self.loss, self.log_p_x_given_z, self.log_q_z_given_x, self.log_p_z, self.global_step],
                              {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

            val_loss += _loss
            pxz += np.mean(_pxz)
            pz += np.mean(_pz)
            qzx += np.mean(_qzx)

        val_loss /= n_val_batches
        pxz /= n_val_batches
        pz /= n_val_batches
        qzx /= n_val_batches

        summary = tf.Summary()
        summary.value.add(tag="Evaluation/loss", simple_value=val_loss)
        summary.value.add(tag="Evaluation/pxz", simple_value=pxz)
        summary.value.add(tag="Evaluation/qzx", simple_value=qzx)
        summary.value.add(tag="Evaluation/pz", simple_value=pz)

        self.val_writer.add_summary(summary, _step)
        self.val_writer.flush()

        x_batch = self.X[self.batch_pointer: self.batch_pointer + batch_size, :]
        s_batch = self.S[self.batch_pointer: self.batch_pointer + batch_size, :]

        _step, _summaries= \
            self.sess.run([self.global_step, self.summaries],
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

        self.train_writer.add_summary(_summaries, _step)
        self.train_writer.flush()

        return val_loss

    def get_llh_estimate(self, Xtest, n_samples=100):
        x_batch = Xtest
        s_batch = (~np.isnan(Xtest)).astype(np.float32)

        _llh = self.sess.run(self.MIWAE,
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: n_samples})

        return _llh

    def tick_batch_pointer(self, batch_size):
        self.batch_pointer += batch_size
        if self.batch_pointer >= self.n - batch_size:
            self.batch_pointer = 0

            try:
                p = np.random.permutation(self.n)
                self.X = self.X[p, :]
                self.S = self.S[p, :]
            except MemoryError as error:
                print("Memory error: no shuffling this time")
                print(error)
            except Exception as exception:
                print("Unexpected exception")
                print(exception)