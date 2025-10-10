.. _Delta:

====================
Delta updates
====================
We present here how TAGI efficiently leverage the Gaussian conditionnal update equations by relying on ``delta_mu[]`` and ``delta_var[]`` in order to only compute the change require to the hidden states without requiring to explicitely compute them. We cover the working principles through two exampple, a first one on the backward step of the linear layer, and a second one for the output updater.

Example through the linear layer hidden-states updates
======================================================

Let's take the backward step used to update the expected for the hidden states from the linear layer,
i.e., ``linear_bwd_fc_delta_z()`` from ``linear_layer.cpp``:

.. code-block:: cpp

   {
       int ni = input_size;
       int no = output_size;
       for (int j = start_chunk; j < end_chunk; j++) {
           int row = j / B;
           int col = j % B;
           float sum_mu_z = 0.0f;
           float sum_var_z = 0.0f;
           for (int i = 0; i < no; i++) {
               sum_mu_z += mu_w[ni * i + row] * delta_mu[col * no + i];
               sum_var_z += mu_w[ni * i + row] * delta_var[col * no + i] * mu_w[ni * i + row];
           }
           // NOTE: Compute directly innovation vector
           delta_mu_z[col * ni + row] = sum_mu_z * jcb[col * ni + row];
           delta_var_z[col * ni + row] = sum_var_z * jcb[col * ni + row] * jcb[col * ni + row];
       }
   }

From the original TAGI paper, we have for the mean RTS update equations:

.. math::

   \begin{aligned}
   f(\mathbf{z}\mid\mathbf{y})
     &= \mathcal{N}\!\big(\mathbf{z};\,\boldsymbol{\mu}_{\mathbf{Z}\mid\mathbf{y}},\,
         \boldsymbol{\Sigma}_{\mathbf{Z}\mid\mathbf{y}}\big) \\[6pt]
   \boldsymbol{\mu}_{\mathbf{Z}\mid\mathbf{y}}
     &= \boldsymbol{\mu}_{\mathbf{Z}}
        + \boldsymbol{\Sigma}_{\mathbf{Z}\mathbf{Z}^{+}}
          \underbrace{\boldsymbol{\Sigma}_{\mathbf{Z}^{+}}^{-1}
          \big(\boldsymbol{\mu}_{\mathbf{Z}^{+}\mid\mathbf{y}}
          - \boldsymbol{\mu}_{\mathbf{Z}^{+}}\big)}_{\mathtt{delta\_mu[.]}} \\[10pt]
     &= \boldsymbol{\mu}_{\mathbf{Z}}
        + \sigma_{Z_i}^{2} \cdot
          \underbrace{\underbrace{\tfrac{da_i}{dz_i}}_{\mathtt{jcb}[.]}\cdot
          \underbrace{\mu_{W_{ij}}}_{\mathtt{mu\_w[.]}}\cdot
          \overbrace{\boldsymbol{\Sigma}_{\mathbf{Z}^{+}}^{-1}
          \big(\boldsymbol{\mu}_{\mathbf{Z}^{+}\mid\mathbf{y}}
          - \boldsymbol{\mu}_{\mathbf{Z}^{+}}\big)}^{\mathtt{delta\_mu[.]}}}
          _{\mathtt{delta\_mu\_z[.]}} \, .
   \end{aligned}

where

.. math::

   \big[\boldsymbol{\Sigma}_{\mathbf{Z}\mathbf{Z}^{+}}\big]_{ij}
   = \sigma_{Z_i}^{2}\cdot
     \underbrace{\tfrac{da_i}{dz_i}}_{\mathtt{jcb}[.]}\cdot
     \underbrace{\mu_{W_{ij}}}_{\mathtt{mu\_w[.]}} \, .

Therefore, by omitting the multiplication by :math:`\sigma_{Z_i}^{2}`,
:math:`\mathtt{delta\_mu\_z[.]}` (which becomes :math:`\mathtt{delta\_mu[.]}` for the
subsequent layer during the backward pass) is already pre-divided by
:math:`(\sigma^{+}_{Z_i})^{2}`.

Example through the output hidden-state update
==============================================

Let's now take the backward step used to update the expected for the hidden states from the output layer,
i.e., ``compute_delta_z_output()`` from ``base_output_updater.cpp``:

.. code-block:: cpp

   {
       float zero_pad = 0;
       float tmp = 0;
       // We compute directly the innovation vector for output layer
       for (int col = start_chunk; col < end_chunk; col++) {
           tmp = jcb[col] / (var_a[col] + var_obs[col]);
           if (isinf(tmp) || isnan(tmp)) {
               delta_mu[col] = zero_pad;
               delta_var[col] = zero_pad;
           } else {
               delta_mu[col] = tmp * (obs[col] - mu_a[col]);
               delta_var[col] = -tmp * jcb[col];
           }
       }
   }

The corresponding update for the expected value reads:

.. math::

   \begin{aligned}
   f\!\big(\mathbf{z}^{(\mathrm{O})}\mid \mathbf{y}\big)
     &= \mathcal{N}\!\Big(\mathbf{z}^{(\mathrm{O})};\,
         \boldsymbol{\mu}_{\mathbf{Z}^{(\mathrm{O})}\mid \mathbf{y}},\,
         \boldsymbol{\Sigma}_{\mathbf{Z}^{(\mathrm{O})}\mid \mathbf{y}}\Big) \\[8pt]
   \boldsymbol{\mu}_{\mathbf{Z}^{(\mathrm{O})}\mid \mathbf{y}}
     &= \boldsymbol{\mu}_{\mathbf{Z}^{(\mathrm{O})}}
        + \mathbf{\Sigma}_{\mathbf{Y}\mathbf{Z}^{(\mathrm{O})}}^{\!\top}\,
          \mathbf{\Sigma}_{\mathbf{Y}}^{-1}\,
          \big(\mathbf{y} - \boldsymbol{\mu}_{\mathbf{Y}}\big) \\[10pt]
     &= \mu_{Z_i^{(\mathrm{O})}}
        + \sigma_{Z^{(\mathrm{O})}_i}^{2}\cdot
          \underbrace{\underbrace{\tfrac{da_i}{dz^{(\mathrm{O})}_i}}_{\mathtt{jcb}[.]}\cdot
          \big(\underbrace{\sigma^{2}_{Z^{(\mathrm{O})}_i}}_{\equiv \sigma_A^2}
          + \sigma_V^{2}\big)^{-1}\,
          (y - \mu_{Y})}_{\mathtt{delta\_mu[.]}} \, .
   \end{aligned}
