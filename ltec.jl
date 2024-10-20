# Import necessary libraries
using Pkg
Pkg.activate(".")
using Plots
using LinearAlgebra  # For SVD

# Define parameters
zstar = 0.0           # Parameter z*
num_steps = 5000      # Number of time steps
h = 0.01              # Time step size
total_time = num_steps * h

# Initialize arrays to store time and (x + z)
time = zeros(num_steps + 1)
x_plus_z = zeros(num_steps + 1)

# Set initial conditions
x0 = 1.0             # Initial condition for x
y0 = 0.0             # Initial condition for y
z0 = 1.0             # Initial condition for z

# Initialize variables
x = x0
y = y0
z = z0

# Store initial (x + z)
time[1] = 0.0
x_plus_z[1] = x + z

# Euler Integration
for i in 1:num_steps
    # Compute derivatives
    dx_dt = -y
    dy_dt = x
    dz_dt = zstar - z  # Since zstar = 0, dz_dt = -z
    
    # Update variables using Euler's method
    x_new = x + h * dx_dt
    y_new = y + h * dy_dt
    z_new = z + h * dz_dt
    
    # Update time
    time[i + 1] = time[i] + h
    
    # Compute (x + z) and store
    x_plus_z[i + 1] = x_new + z_new
    
    # Update variables for next iteration
    x, y, z = x_new, y_new, z_new
end

# Plot the time series of (x + z)(t)
plot(time, x_plus_z, title="Time Series of (x + z)(t)",
     xlabel="Time", ylabel="x + z",
     legend=false, linewidth=2)

# Optional: Save the time series plot to a file
# savefig("x_plus_z_timeseries.png")

###############################
# Additional Code Starts Here #
###############################

# Parameters for Hankel Matrix
embedding_dim = 100  # Number of columns (embedding dimension)

# Function to create a Hankel matrix from a time series
function create_hankel_matrix(ts::Vector{Float64}, d::Int)
    n = length(ts)
    m = n - d + 1
    if m <= 0
        error("Embedding dimension d is too large for the time series length.")
    end
    # Initialize the Hankel matrix
    H = zeros(Float64, m, d)
    for i in 1:m
        H[i, :] = ts[i:i+d-1]
    end
    return H
end

# Create Hankel matrix from x_plus_z
H = create_hankel_matrix(x_plus_z, embedding_dim)

# Compute Singular Value Decomposition (SVD)
U, S, V = svd(H)

# Compute squared singular values
squared_singular_values = S.^2

# Sort the squared singular values in descending order
sorted_squared_singular_values = sort(squared_singular_values, rev=true)

# Generate indices for the scatter plot
indices = 1:length(sorted_squared_singular_values)

# Plot the squared singular values as a scatter plot
scatter(indices, sorted_squared_singular_values,
       title="Squared Singular Values of Hankel Matrix",
       xlabel="Index",
       ylabel="Squared Singular Value",
       legend=false,
       markersize=4,
       markercolor=:blue)

# Optional: Save the singular values scatter plot
# savefig("singular_values_squared_scatter.png")

###############################
# Singular Value Component Extraction #
###############################

# Function to reconstruct time series from a Hankel matrix
function reconstruct_time_series_from_hankel(H_reconstructed::Matrix{Float64})
    m, d = size(H_reconstructed)
    n = m + d - 1
    ts_reconstructed = zeros(Float64, n)
    counts = zeros(Int, n)
    
    for i in 1:m
        for j in 1:d
            ts_reconstructed[i + j - 1] += H_reconstructed[i, j]
            counts[i + j - 1] += 1
        end
    end
    
    return ts_reconstructed ./ counts
end

# Function to extract and plot a specific singular value component
function extract_and_plot_singular_component(U, S, V, embedding_dim, selected_sv_index::Int, h::Float64)
    if selected_sv_index < 1 || selected_sv_index > embedding_dim
        error("Selected singular value index must be between 1 and embedding_dim ($embedding_dim).")
    end
    
    # Create a truncated singular value vector: only keep the selected singular value
    S_truncated = zeros(length(S))
    S_truncated[selected_sv_index] = S[selected_sv_index]
    
    # Reconstruct the Hankel matrix using the truncated SVD
    H_truncated = U * Diagonal(S_truncated) * V'
    
    # Reconstruct the time series from the truncated Hankel matrix
    ts_component = reconstruct_time_series_from_hankel(H_truncated)
    
    # Define the time vector for the reconstructed component
    # Since the original time vector has n = m + d -1 points
    m, d = size(H_truncated)
    n = m + d - 1
    ts_time = range(0, step=h, length=n)
    
    # Plot the reconstructed time series component
    plt = plot(ts_time, ts_component,
      title="Reconstructed Time Series Component from Singular Value $selected_sv_index",
      xlabel="Time",
      ylabel="Component Value",
      legend=false,
      linewidth=2,
      color=:red
    )
    
    # Optional: Save the reconstructed component plot
    # savefig("singular_value_${selected_sv_index}_component.png")
    
    return ts_component, ts_time, plt
end

# Example Usage:
# Select a singular value index to extract its corresponding time series component
selected_sv_index = 3  # Change this value between 1 and embedding_dim as needed

# Extract and plot the singular value component
ts_component, ts_time, plt = extract_and_plot_singular_component(U, S, V, embedding_dim, selected_sv_index, h)
display(plt)

# Optional: Overlay the reconstructed component on the original time series for comparison
# plot(time[1:length(ts_time)], x_plus_z[1:length(ts_time)],
#      label="Original (x + z)(t)", linewidth=1, color=:blue)
# plot!(ts_time, ts_component, label="Singular Value $selected_sv_index Component", linewidth=2, color=:red)

###############################
# Recursive Hankel-SVD Decomposition #
###############################

# Function to perform Hankel-SVD decomposition and plot squared singular values
function decompose_and_plot_svd(ts::Vector{Float64}, embedding_dim::Int, h::Float64, component_index::Int)
    # Create Hankel matrix from the reconstructed time series component
    H_new = create_hankel_matrix(ts, embedding_dim)
    
    # Compute SVD of the new Hankel matrix
    U_new, S_new, V_new = svd(H_new)
    
    # Compute squared singular values
    squared_singular_values_new = S_new.^2
    
    # Sort the squared singular values in descending order
    sorted_squared_singular_values_new = sort(squared_singular_values_new, rev=true)
    
    # Generate indices for the scatter plot
    indices_new = 1:length(sorted_squared_singular_values_new)
    
    # Plot the squared singular values as a scatter plot
    scatter(indices_new, sorted_squared_singular_values_new,
           title="Squared Singular Values of Re-decomposed Hankel Matrix (Component $component_index)",
           xlabel="Index",
           ylabel="Squared Singular Value",
           legend=false,
           markersize=4,
           markercolor=:green)
    
    # Optional: Save the new singular values scatter plot
    # savefig("singular_values_squared_scatter_component_${component_index}.png")
end

# Perform Hankel-SVD decomposition on the reconstructed singular component
decompose_and_plot_svd(ts_component, embedding_dim, h, selected_sv_index)
