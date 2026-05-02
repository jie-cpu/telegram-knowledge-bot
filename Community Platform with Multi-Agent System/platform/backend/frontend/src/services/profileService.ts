import { Profile, ProfileCreate, ProfileUpdate } from '../types/profile';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Fetches a profile by user ID.
 * @param userId - The ID of the user whose profile to fetch.
 * @returns The profile data.
 */
export const getProfile = async (userId: number): Promise<Profile> => {
  const response = await fetch(`${API_BASE_URL}/api/profiles/${userId}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch profile: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Creates a new profile.
 * @param profileData - The profile data to create.
 * @returns The created profile.
 */
export const createProfile = async (profileData: ProfileCreate): Promise<Profile> => {
  const response = await fetch(`${API_BASE_URL}/api/profiles`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(profileData),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to create profile: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Updates an existing profile.
 * @param userId - The ID of the user whose profile to update.
 * @param profileData - The profile data to update.
 * @returns The updated profile.
 */
export const updateProfile = async (userId: number, profileData: ProfileUpdate): Promise<Profile> => {
  const response = await fetch(`${API_BASE_URL}/api/profiles/${userId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(profileData),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to update profile: ${response.statusText}`);
  }

  return response.json();
};
