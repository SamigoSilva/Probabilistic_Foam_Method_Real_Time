#include <vector>
#include <array>
#include <cmath>
#include <omp.h>

// Declaração forward da função original
void original_update_function(
    std::vector<std::vector<float>>& grid,
    const std::vector<std::array<float, 2>>& obstacles,
    float robot_x, float robot_y);

// Implementação da função original
void original_update_function(
    std::vector<std::vector<float>>& grid,
    const std::vector<std::array<float, 2>>& obstacles,
    float robot_x, float robot_y) 
{
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < grid.size(); x++) {
        for (size_t y = 0; y < grid[0].size(); y++) {
            float repulsive = 0.0f;
            for (const auto& obs : obstacles) {
                float dx = x - obs[0];
                float dy = y - obs[1];
                float dist_sq = dx*dx + dy*dy + 1e-5f;
                repulsive += 1.0f / dist_sq;
            }
            
            float dx = x - robot_x;
            float dy = y - robot_y;
            float attractive = 0.1f * std::hypot(dx, dy);
            
            grid[x][y] = repulsive + attractive;
        }
    }
}

// Função wrapper para interface C
extern "C" {
    __declspec(dllexport) void update_potential_field(
        float* grid_data, int grid_width, int grid_height,
        const float* obstacles, int num_obstacles,
        float robot_x, float robot_y)
    {
        // Converter para estruturas C++
        std::vector<std::vector<float>> grid(grid_width, std::vector<float>(grid_height));
        std::vector<std::array<float, 2>> obstacles_vec(num_obstacles);
        
        // Preencher grid
        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                grid[x][y] = grid_data[x * grid_height + y];
            }
        }
        
        // Preencher obstáculos
        for (int i = 0; i < num_obstacles; i++) {
            obstacles_vec[i][0] = obstacles[i*2];
            obstacles_vec[i][1] = obstacles[i*2 + 1];
        }
        
        // Chamar função original
        original_update_function(grid, obstacles_vec, robot_x, robot_y);
        
        // Copiar resultados de volta
        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                grid_data[x * grid_height + y] = grid[x][y];
            }
        }
    }
}