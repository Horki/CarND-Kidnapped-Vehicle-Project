/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[3]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  particles.reserve(num_particles);
  weights.reserve(num_particles);
  // Create a normal Gaussian distribution
  std::normal_distribution<double> dist_x(x,         std[0]);
  std::normal_distribution<double> dist_y(y,         std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (size_t i = 0; i < num_particles; ++i) {
    particles.push_back(Particle {
      .id     = int(i),
      .x      = dist_x(gen),
      .y      = dist_y(gen),
      .theta  = dist_theta(gen),
      .weight = 1.0
    });
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t,  double std_pos[3],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // Create a normal Gaussian distribution
  std::normal_distribution<double> dist_x(0.0,     std_pos[0]);
  std::normal_distribution<double> dist_y(0.0,     std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

  std::transform(particles.begin(), particles.end(), particles.begin(),
                 [&](Particle & p) -> Particle {
    // Add measurements
    if (std::fabs(yaw_rate) < 0.00001) {
      p.x += std::cos(p.theta) * velocity * delta_t;
      p.y += std::sin(p.theta) * velocity * delta_t;
    } else {
      p.x += (velocity / yaw_rate) *
        (std::sin(p.theta + (yaw_rate * delta_t)) - std::sin(p.theta));
      p.y += (velocity / yaw_rate) *
        (std::cos(p.theta) - std::cos(p.theta + (yaw_rate * delta_t)));
      p.theta += (yaw_rate * delta_t);
    }
    // Add random Gaussian nose.
    p.x     += dist_x(gen);
    p.y     += dist_y(gen);
    p.theta += dist_theta(gen);
    return p;
  });

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[2],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double min_x =  std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y =  std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();

  for (const auto& part : particles) {
    min_x = std::min(min_x, part.x);
    max_x = std::max(max_x, part.x);
    min_y = std::min(min_y, part.y);
    max_y = std::max(max_y, part.y);
  }

  std::vector<Map::single_landmark_s> landmarks;
  for (const auto& land : map_landmarks.landmark_list) {
    if (land.x_f < min_x - sensor_range || land.x_f > max_x + sensor_range
      || land.y_f < min_y - sensor_range || land.y_f > max_y + sensor_range) {
      continue;
    }
    landmarks.push_back(land);
  }
  double sigma_x2 = std::pow(std_landmark[0], 2);
  double sigma_y2 = std::pow(std_landmark[1], 2);

  for (auto& part : particles) {
    // reset particle weight
    part.weight = 1;

    for (auto& obs : observations) {
      double xabs = std::cos(part.theta) * obs.x - std::sin(part.theta) * obs.y + part.x;
      double yabs = std::sin(part.theta) * obs.x + std::cos(part.theta) * obs.y + part.y;
      double min_norm_dist_sq = std::numeric_limits<double>::infinity();
      for (auto& land : landmarks) {
        double dx = xabs - land.x_f;
        double dy = yabs - land.y_f;
        double norm_dist_sq = dx * dx / sigma_x2 + dy * dy / sigma_y2;
        min_norm_dist_sq = std::min(min_norm_dist_sq, norm_dist_sq);
      }

      double prob = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1])
                    * std::exp(-0.5 * min_norm_dist_sq);
      part.weight *= prob;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<double> w;
  for (auto& part : particles) {
    w.push_back(part.weight);
  }
  std::discrete_distribution<int> dd(w.begin(), w.end());
  std::vector<Particle> new_particles(num_particles);
  for (size_t i = 0; i < num_particles; ++i) {
    new_particles.push_back(particles[dd(gen)]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x      = sense_x;
  particle.sense_y      = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v = (coord == "X") ? best.sense_x : best.sense_y;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
