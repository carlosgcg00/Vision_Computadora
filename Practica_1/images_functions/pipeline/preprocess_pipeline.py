import matplotlib.pyplot as plt
import math

def pipeline_preprocess_img(img, steps, plot_flag=True, save_plot=None):
    """
    Sequentially applies a set of preprocessing functions to an image
    and optionally displays or saves a subplot with all intermediate results.

    Args:
        img (numpy.ndarray): Input image.
        steps (list): List of tuples (function, title, params_dict).
                      Each tuple defines one step in the pipeline.
        plot_flag (bool): If True, displays the subplot with all intermediate results.
        save_plot (str or None): If provided, saves the subplot to this path.
                                 If None, the figure is not saved.

    Returns:
        result (numpy.ndarray): Final processed image.
        intermediates (list): List containing intermediate results and metadata.
        fig (matplotlib.figure.Figure or None): The matplotlib figure, if generated.
    """
    result = img
    intermediates = [{
        "step": 0,
        "func_title": "Original Image",
        "func": "input",
        "params": {},
        "result": img
    }]

    # Run each step in sequence
    for i, (func, func_title, params) in enumerate(steps, start=1):
        params = params.copy()
        # Disable internal plotting if the function supports it
        if "plot_flag" in params:
            params["plot_flag"] = False

        result = func(result, **params)

        intermediates.append({
            "step": i,
            "func_title": func_title,
            "func": func.__name__,
            "params": params,
            "result": result
        })

    # Plot and/or save results
    fig = None
    if plot_flag or save_plot:
        n = len(intermediates)
        max_cols = 4
        n_cols = min(n, max_cols)
        n_rows = math.ceil(n / n_cols)

        fig_width = 5 * n_cols
        fig_height = 4 * n_rows

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axs = axs.flatten()

        for idx, step in enumerate(intermediates):
            ax = axs[idx]
            ax.imshow(step["result"], cmap='gray')
            title = f"{idx}-{step['step']}: {step['func_title']}"
            ax.set_title(title)
            ax.axis('off')

        # Ocultar los ejes sobrantes si no se llenan todas las celdas
        for j in range(len(intermediates), len(axs)):
            axs[j].axis('off')

        plt.tight_layout()

        # Save subplot if a path is provided
        if save_plot:
            plt.savefig(save_plot, bbox_inches='tight')

        if plot_flag:
            plt.show()
        else:
            plt.close(fig)

    return result, intermediates
