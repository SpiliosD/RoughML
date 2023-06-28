import logging
import time

import torch

from roughml.data.model_weights import model_weights

logger = logging.getLogger(__name__)


# Returns a normalized and a normalized weighted value of a measure. Normalization occurs before weighting.
# The function returns both the above value, but also the new max value so far.
def normalizedAndWeightedLoss(value, weight, max_value_so_far):
    # Update maximum
    if max_value_so_far < value:
        max_value_so_far = value
    # normalize value
    value_norm = value / max_value_so_far
    # normalize weighted value
    value_norm_weighted = weight * value / max_value_so_far

    # Return normalized value, normalized weighted value, and max value so far
    return value_norm, value_norm_weighted, max_value_so_far


"""
The training of the GAN takes place in this function. It closely follows Algorithm 1 from Goodfellow’s paper, 
while abiding by some of the best practices shown in ganhacks. Namely, we will “construct different mini-batches 
for real and fake” images, and also adjust Generator’s objective function to maximize logD(G(z)). 
Training is split up into two main parts. Part 1 updates the Discriminator and Part 2 updates the Generator.
Furthermore, custom losses are calculated for the generator overall loss. Specifically content loss and vector content 
loss are general terms whereas in our case content loss is N-Gram graph loss and vector content loss is 
Height Histogram and Fourier loss. Each independent loss is normalized and weighted Those custom losses are accumulated 
to generators bce loss.
"""

def per_epoch(
    generator,
    discriminator,
    dataloader,
    optimizer_generator,
    optimizer_discriminator,
    criterion,
    content_loss_fn=None,
    vector_content_loss_fn=None,
    loss_weights=[1.0, 1.0, 1.0],
    losses_maxima={},
    losses_raw={},
    log_every_n=None,
    load_checkpoint=None,
):
    generator.train()

    # clear loss values before next batch
    losses_raw = dict.fromkeys(losses_raw, 0)
    generator_loss, discriminator_loss = 0, 0
    dis_out_real_batch_real_label, dis_out_gen_batch_fake_label, dis_out_gen_batch_real_label = 0, 0, 0
    BCELoss, NGramGraphLoss, HeightHistogramAndFourierLoss = 0, 0, 0

    start_time = time.time()

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # For each batch in the dataloader
    for train_iteration, X_batch in enumerate(dataloader):
        # change batch type to match model's checkpoint weights when model is loaded to prevent runtime error
        if load_checkpoint:
            X_batch = X_batch.float()
        """
        Part 1 - Train the Discriminator
        Recall, the goal of training the discriminator is to maximize the probability of correctly classifying a given 
        input as real or fake. In terms of Goodfellow, we wish to “update the discriminator by ascending its 
        stochastic gradient”. Practically, we want to maximize log(D(x))+log(1−D(G(z))). Due to the separate mini-batch 
        suggestion from ganhacks, we will calculate this in two steps. First, we will construct a batch of real samples 
        from the training set, forward pass through D, calculate the loss log(D(x)), then calculate the gradients in a 
        backward pass. Secondly, we will construct a batch of fake samples with the current generator, forward pass 
        this batch through D, calculate the loss log(1−D(G(z))),  and accumulate the gradients with a backward pass.
        Now, with the gradients accumulated from both the all-real and all-fake batches, we call a step of the 
        Discriminator’s optimizer.
        """
        # (Part 1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        # Train with all-real batch
        # Set the gradients of Discriminator to zero
        discriminator.zero_grad()
        # Format batch on real images with real label
        batch_size = X_batch.size(0)
        label = torch.full(
            (batch_size,), real_label, dtype=X_batch.dtype, device=X_batch.device
        )
        # Forward pass real batch through Discriminator
        output = discriminator(X_batch).view(-1)
        # Calculate loss on all-real batch
        discriminator_error_real = criterion(output, label)
        # save raw discriminator BCE loss for real images to view in log file
        losses_raw['raw_dis_bce_loss_real'] += discriminator_error_real.item()
        # normalize discriminator BCE loss on real images
        if losses_maxima['max_dis_bce_loss_real'] < discriminator_error_real.item():
            losses_maxima['max_dis_bce_loss_real'] = discriminator_error_real.item()
        discriminator_error_real /= losses_maxima['max_dis_bce_loss_real']
        # Calculate gradients for Discriminator for real images in backward pass
        discriminator_error_real.backward()
        # Average Discriminator output on batch during forward pass on real images with real label (D(x))
        dis_out_real_batch_real_lbl = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(
            batch_size,
            *generator.feature_dims,
            dtype=X_batch.dtype,
            device=X_batch.device,
        )
        # Generate fake image batch with Generator
        fake = generator(noise)
        # label for fake/generated images is 0
        label.fill_(fake_label)
        # Classify all fake batch with Discriminator
        output = discriminator(fake.detach()).view(-1)
        # Calculate Discriminator's loss on the all-fake batch
        discriminator_error_fake = criterion(output, label)
        # save raw discriminator BCE loss for fake/generated images to view in log file
        losses_raw['raw_dis_bce_loss_fake'] += discriminator_error_fake.item()
        # normalize discriminator BCE loss on fake images
        if losses_maxima['max_dis_bce_loss_fake'] < discriminator_error_fake.item():
            losses_maxima['max_dis_bce_loss_fake'] = discriminator_error_fake.item()
        discriminator_error_fake /= losses_maxima['max_dis_bce_loss_fake']
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        discriminator_error_fake.backward()
        dis_out_gen_batch_fake_lbl = output.mean().item()
        # Compute error of Discriminator as sum over the fake and the real batches
        discriminator_error_total = (discriminator_error_real.item() + discriminator_error_fake.item()) / 2
        # Update Discriminator
        optimizer_discriminator.step()

        """
        Part 2 - Train the Generator
        As stated in the original paper, we want to train the Generator by minimizing log(1−D(G(z))) in an effort to 
        generate better fakes. As mentioned, this was shown by Goodfellow to not provide sufficient gradients, 
        especially early in the learning process. As a fix, we instead wish to maximize log(D(G(z))).
        In the code we accomplish this by: classifying the Generator output from Part 1 with the Discriminator, 
        computing G’s loss using real labels as GT, computing G’s gradients in a backward pass, and finally updating 
        G’s parameters with an optimizer step. It may seem counter-intuitive to use the real labels as GT labels for 
        the loss function, but this allows us to use the log(x) part of the BCELoss (rather than the log(1−x) part) 
        which is exactly what we want.
        """
        # (Part 2) Update G network: maximize log(D(G(z)))

        # Set the gradients of Generator to zero
        generator.zero_grad()
        # fake labels are real for generator cost
        label.fill_(real_label)
        # Since we just updated Discriminator, perform another forward pass of all-fake batch through Discriminator
        output = discriminator(fake).view(-1)

        # Calculate Generator's loss
        # For each loss component
        # Normalize it based on the maximum value we have seen in this loss for this batch
        # Weight it and add it to the overall loss

        # Generator BCE loss calculated as log(D(G(z)))
        generator_bce_loss = criterion(output, label)  # get generator BCE loss for this batch
        # save raw generator BCE loss for fake/generated images to view in log file
        losses_raw['raw_gen_bce_loss'] += generator_bce_loss.item()
        # Calculate the normalized weighted value and also get the new maximum
        if losses_maxima['max_gen_bce_loss'] < generator_bce_loss.item():
            losses_maxima['max_gen_bce_loss'] = generator_bce_loss.item()
        # normalize generator bce loss
        generator_bce_loss /= losses_maxima['max_gen_bce_loss']
        # assign normalized value to another var for plotting
        norm_gen_bce_loss = generator_bce_loss.item()
        # weight generator bce loss
        generator_bce_loss *= loss_weights[0]

        # Generator content loss (N-Gram graph loss)
        generator_content_loss = torch.mean(content_loss_fn(fake.cpu().detach().numpy().squeeze())).to(fake.device)  # Get generator content-based-loss mean on fake images batch
        # save raw generator content loss for fake/generated images to view in log file
        losses_raw['raw_gen_NGramGraphLoss'] += generator_content_loss.item()
        # Calculate the normalized weighted value and also get the new maximum
        if losses_maxima['max_gen_NGramGraphLoss'] < generator_content_loss.item():
            losses_maxima['max_gen_NGramGraphLoss'] = generator_content_loss.item()
        # normalize generator content loss
        generator_content_loss /= losses_maxima['max_gen_NGramGraphLoss']
        # assign normalized value to another var for plotting
        norm_gen_content_loss = generator_content_loss.item()
        # weight generator content loss
        generator_content_loss *= loss_weights[1]
        
        # Generator vector content loss (Height Histogram and Fourier loss)
        generator_vector_content_loss = torch.mean(vector_content_loss_fn(fake.cpu().detach().numpy().squeeze())).to(fake.device) # get mean generator height histogram and fourier loss on fake images batch
        # save raw generator vector content loss for fake/generated images to view in log file
        losses_raw['raw_gen_HeightHistogramAndFourierLoss'] += generator_vector_content_loss.item()
        # Calculate the normalized weighted value and also get the new maximum
        if losses_maxima['max_gen_HeightHistogramAndFourierLoss'] < generator_vector_content_loss.item():
            losses_maxima['max_gen_HeightHistogramAndFourierLoss'] = generator_vector_content_loss.item()
        # normalize generator vector content loss
        generator_vector_content_loss /= losses_maxima['max_gen_HeightHistogramAndFourierLoss']
        # assign to another var for plotting
        norm_gen_hist_fourier_loss = generator_vector_content_loss.item()
        # weight generator vector content loss
        generator_vector_content_loss *= loss_weights[2]

        # Update overall generator loss with batch contribution
        generator_error_total = generator_bce_loss + generator_content_loss + generator_vector_content_loss

        # Calculate gradients for G, which propagate through the discriminator
        generator_error_total.backward()
        dis_out_gen_batch_real_lbl = output.mean().item()
        # Update Generator
        optimizer_generator.step()

        # Display all model layer weights
        # generator_data = model_weights(generator)

        # accumulate losses for this batch
        generator_loss += generator_error_total.item() # update total generator loss for this batch
        BCELoss += norm_gen_bce_loss # update generator Binary Cross-Entropy loss for this batch
        NGramGraphLoss += norm_gen_content_loss # update generator N-Gram Graph loss for this batch
        HeightHistogramAndFourierLoss += norm_gen_hist_fourier_loss # update generator Height Histogram And Fourier loss for this batch
        discriminator_loss += discriminator_error_total # update total discriminator loss for this batch
        dis_out_real_batch_real_label += dis_out_real_batch_real_lbl # update discriminator output for real images for this batch
        dis_out_gen_batch_fake_label += dis_out_gen_batch_fake_lbl # update discriminator output for generated images for this batch
        dis_out_gen_batch_real_label += dis_out_gen_batch_real_lbl # update discriminator output for generated images for this batch

        if log_every_n is not None and not train_iteration % log_every_n:
            logger.info(
                "Minibatch iteration:%02d with batch size:%02d ended after %7.3f seconds",
                train_iteration,
                batch_size,
                time.time() - start_time,
            )
            start_time = time.time()


    # calculate average losses over an epoch
    generator_loss /= len(dataloader)
    BCELoss /= len(dataloader)
    NGramGraphLoss /= len(dataloader)
    HeightHistogramAndFourierLoss /= len(dataloader)
    discriminator_loss /= len(dataloader)
    dis_out_real_batch_real_label /= len(dataloader)
    dis_out_gen_batch_fake_label /= len(dataloader)
    dis_out_gen_batch_real_label /= len(dataloader)
    for elem in losses_raw.keys():
        losses_raw[elem] = losses_raw.get(elem) / len(dataloader)

    return (
        generator_loss,
        discriminator_loss,
        dis_out_real_batch_real_label,
        dis_out_gen_batch_fake_label,
        dis_out_gen_batch_real_label,
        NGramGraphLoss,
        HeightHistogramAndFourierLoss,
        BCELoss, 
        losses_maxima,
        losses_raw,
    )
