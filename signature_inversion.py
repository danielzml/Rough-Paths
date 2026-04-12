import numpy as np
import matplotlib.pyplot as plt
from signature import signatures_via_cde_batch


def spherical_random_search(
    sig_target_np,
    n_steps,
    spatial_dim,
    depth,
    n_iter,
    b,
    eps,
    eps_decay,
    original_spatial=None,
    save_interval=10
):
    sig_target = sig_target_np
   
    # Initialize
    spatial = np.cumsum(np.random.randn(n_steps, spatial_dim), axis=0)
    
    # Align initial spatial if original provided (FIX #2)
    if original_spatial is not None:
        spatial = spatial + (original_spatial[0] - spatial[0])
    
    time = np.arange(n_steps).reshape(-1, 1)
   
    # Evaluate initial
    path_current = np.concatenate([time, spatial], axis=1)
    path_batch = path_current.reshape(1, n_steps, spatial_dim + 1)
    sig_current = signatures_via_cde_batch(path_batch, depth)[0]
    loss_current = np.linalg.norm(sig_current - sig_target)
    losses = [loss_current]
   
    for it in range(n_iter):

        # Generate random directions
        directions = np.random.randn(b, n_steps, spatial_dim)
        norms = np.linalg.norm(directions.reshape(b, -1), axis=1, keepdims=True)
        directions = directions / norms.reshape(b, 1, 1)
       
        # Align candidates before computing signatures
        candidates = spatial + eps * directions
        
        # Align all candidates to start at original_spatial[0]
        if original_spatial is not None:
            start_offsets = candidates[:, 0, :] - original_spatial[0]  # (b, spatial_dim)
            candidates = candidates - start_offsets[:, np.newaxis, :]  # Align all
        
        time_batch = np.broadcast_to(time.reshape(1, n_steps, 1), (b, n_steps, 1))
        paths_batch = np.concatenate([time_batch, candidates], axis=2)
       
        # Batch compute signatures 
        sigs_batch = signatures_via_cde_batch(paths_batch, depth)
        losses_batch = np.linalg.norm(sigs_batch - sig_target, axis=1)
       
        best_idx = np.argmin(losses_batch)
        best_loss = losses_batch[best_idx]
       
        if best_loss < loss_current:
            spatial = candidates[best_idx]  # Already aligned
            loss_current = best_loss
       
        losses.append(loss_current)
        eps *= eps_decay
       
        if it % save_interval == 0 or it == n_iter - 1:
            print(f"Iter {it}: Loss = {loss_current:.6f}, eps = {eps:.4f}")
           
            if original_spatial is not None:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

                # Plot paths 
                ax.plot(original_spatial[:,0], original_spatial[:,1], original_spatial[:,2],
                        label="Original", color="blue", linewidth=2, alpha=0.8)
                ax.plot(spatial[:,0], spatial[:,1], spatial[:,2],
                        label="Recovered", color="red", linestyle="--", linewidth=2)

                ax.set_xlabel('X1')
                ax.set_ylabel('X2')
                ax.set_zlabel('X3')
                ax.legend()
                ax.set_title(f"Iteration {it} | Loss = {loss_current:.6f}")
                plt.tight_layout()
                plt.savefig(f"reconstruction.png", dpi=150, bbox_inches='tight')
                plt.close()

    return spatial, losses


if __name__ == "__main__":
    n_steps = 40
    spatial_dim = 3
    depth = 4
   
    print("Generating original path...")

    original_spatial = np.zeros((n_steps, spatial_dim))

    scale = 2
    increments = scale * np.random.randn(n_steps - 1, 3)
    bm3d = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])

    original_spatial = bm3d
   
    time = np.arange(n_steps).reshape(-1, 1)
    original_path = np.concatenate([time, original_spatial], axis=1)
   
    print("Computing target signature...")
    target_sig = signatures_via_cde_batch(
        original_path.reshape(1, n_steps, spatial_dim + 1),
        depth
    )[0]
   
    print(f"Target signature shape: {target_sig.shape}")
    print(f"Target signature norm: {np.linalg.norm(target_sig):.4f}")
   
    print("\nStarting inversion...")
    recovered_spatial, losses = spherical_random_search(
        sig_target_np=target_sig,
        n_steps=n_steps,
        spatial_dim=spatial_dim,
        depth=depth,
        n_iter=2000,
        b=50,
        eps=12,
        eps_decay=0.98,
        original_spatial=original_spatial,
        save_interval=5
    )
   
    # Final plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss (L2 norm)')
    axes[0].set_title('Reconstruction Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
   
    ax = fig.add_subplot(122, projection='3d')
    ax.plot(original_spatial[:,0], original_spatial[:,1], original_spatial[:,2],
            'b-', linewidth=3, label='Original', alpha=0.8)
    ax.plot(recovered_spatial[:,0], recovered_spatial[:,1], recovered_spatial[:,2],
            'r--', linewidth=2, label='Recovered')
    ax.scatter(original_spatial[0,0], original_spatial[0,1], original_spatial[0,2],
            c='blue', s=100, marker='o', zorder=5, label='Start')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('3D Path Reconstruction')
    ax.legend()
   
    plt.tight_layout()
    plt.savefig("final_reconstruction.png", dpi=150, bbox_inches='tight')
    plt.close()
   
    print(f"\nFinal loss: {losses[-1]:.6f}")
