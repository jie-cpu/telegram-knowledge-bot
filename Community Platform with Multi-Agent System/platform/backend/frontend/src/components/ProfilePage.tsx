import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getProfile } from '../services/profileService';
import { Profile } from '../types/profile';

const ProfilePage: React.FC = () => {
  const { userId } = useParams<{ userId: string }>();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await getProfile(Number(userId));
        setProfile(data);
      } catch (err: any) {
        setError(err.message || 'Failed to load profile');
      } finally {
        setLoading(false);
      }
    };

    if (userId) {
      fetchProfile();
    }
  }, [userId]);

  if (loading) {
    return <div className="profile-page loading">Loading profile...</div>;
  }

  if (error) {
    return <div className="profile-page error">Error: {error}</div>;
  }

  if (!profile) {
    return <div className="profile-page not-found">Profile not found.</div>;
  }

  return (
    <div className="profile-page">
      <div className="profile-header">
        {profile.avatar_url && (
          <img src={profile.avatar_url} alt={`${profile.name}'s avatar`} className="profile-avatar" />
        )}
        <h1>{profile.name}</h1>
        <p className="profile-bio">{profile.bio || 'No bio provided.'}</p>
        <p className="profile-meta">
          Member since {new Date(profile.created_at).toLocaleDateString()}
        </p>
        <Link to={`/profile/${userId}/edit`} className="btn btn-primary">
          Edit Profile
        </Link>
      </div>
    </div>
  );
};

export default ProfilePage;
