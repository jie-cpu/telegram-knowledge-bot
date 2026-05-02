/**
 * TypeScript types for Profile data.
 */

export interface Profile {
  id: number;
  user_id: number;
  name: string;
  bio: string | null;
  avatar_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProfileCreate {
  user_id: number;
  name: string;
  bio?: string | null;
  avatar_url?: string | null;
}

export interface ProfileUpdate {
  name?: string;
  bio?: string | null;
  avatar_url?: string | null;
}
