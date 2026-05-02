import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createProfile } from '../services/profileService';
import { ProfileCreate } from '../types/profile';

const ProfileCreateForm: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<ProfileCreate>({
    user_id: 0,
    name: '',
    bio: '',
    avatar_url: '',
  });
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const newProfile = await createProfile(formData);
      navigate(`/profile/${newProfile.user_id}`);
    } catch (err: any) {
      setError(err.message || 'Failed to create profile');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="profile-create-form">
      <h2>Create Profile</h2>
      {error && <div className="error-message">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="user_id">User ID</label>
          <input
            type="number"
            id="user_id"
            name="user_id"
            value={formData.user_id}
            onChange={handleChange}
            required
            min={1}
          />
        </div>
        <div className="form-group">
          <label htmlFor="name">Name</label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            required
            maxLength={100}
          />
        </div>
        <div className="form-group">
          <label htmlFor="bio">Bio</label>
          <textarea
            id="bio"
            name="bio"
            value={formData.bio}
            onChange={handleChange}
            maxLength={500}
            rows={4}
          />
        </div>
        <div className="form-group">
          <label htmlFor="avatar_url">Avatar URL</label>
          <input
            type="url"
            id="avatar_url"
            name="avatar_url"
            value={formData.avatar_url}
            onChange={handleChange}
            maxLength={500}
            placeholder="https://example.com/avatar.jpg"
          />
        </div>
        <button type="submit" disabled={submitting} className="btn btn-primary">
          {submitting ? 'Creating...' : 'Create Profile'}
        </button>
      </form>
    </div>
  );
};

export default ProfileCreateForm;
