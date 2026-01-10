"""
Speaker Profile Manager

Manages speaker profiles and embeddings in Neo4j database.
"""
import logging
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerProfileManager:
    """
    Manages speaker profiles in Neo4j database.
    
    Handles CRUD operations for speaker data including
    voice embeddings, metadata, and preferences.
    """
    
    def __init__(self, neo4j_client):
        """
        Initialize the profile manager.
        
        Args:
            neo4j_client: Neo4jClient instance
        """
        self.client = neo4j_client
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create speaker schema if it doesn't exist."""
        try:
            # Create constraint
            constraint_query = """
            CREATE CONSTRAINT speaker_id_unique IF NOT EXISTS
            FOR (s:Speaker) REQUIRE s.speaker_id IS UNIQUE
            """
            self.client.execute_query(constraint_query)
            
            # Create index
            index_query = """
            CREATE INDEX speaker_name_index IF NOT EXISTS
            FOR (s:Speaker) ON (s.name)
            """
            self.client.execute_query(index_query)
            
            logger.info("Speaker schema ensured")
        except Exception as e:
            logger.warning(f"Schema creation warning: {e}")
    
    def create_speaker(
        self,
        speaker_id: str,
        name: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None,
        voice_preferences: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new speaker profile.
        
        Args:
            speaker_id: Unique speaker identifier
            name: Speaker name
            embedding: Voice embedding vector
            metadata: Optional metadata dictionary
            voice_preferences: Optional TTS voice preferences:
                {
                    'voice_name': str,  # TTS voice ID
                    'speed': float,     # Speaking rate (0.5-2.0)
                    'pitch': float      # Pitch shift (-10 to +10)
                }
            
        Returns:
            Created speaker data
        """
        # Convert embedding to list for JSON storage
        embedding_list = embedding.tolist()
        
        # Default voice preferences
        if voice_preferences is None:
            voice_preferences = {
                'voice_name': 'default',
                'speed': 1.0,
                'pitch': 0.0
            }
        
        query = """
        CREATE (s:Speaker {
            speaker_id: $speaker_id,
            name: $name,
            embedding: $embedding,
            voice_preferences: $voice_preferences,
            created_at: datetime(),
            last_seen: datetime(),
            sample_count: 1
        })
        RETURN s
        """
        
        params = {
            'speaker_id': speaker_id,
            'name': name,
            'embedding': embedding_list,
            'voice_preferences': json.dumps(voice_preferences)
        }
        
        try:
            result = self.client.execute_query(query, params)
            logger.info(f"Created speaker profile: {name} ({speaker_id}) with voice: {voice_preferences['voice_name']}")
            return {
                'speaker_id': speaker_id,
                'name': name,
                'embedding_dim': len(embedding_list),
                'voice_preferences': voice_preferences
            }
        except Exception as e:
            logger.error(f"Failed to create speaker: {e}")
            raise
    
    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        """
        Get speaker profile by ID.
        
        Args:
            speaker_id: Speaker identifier
            
        Returns:
            Speaker data or None if not found
        """
        query = """
        MATCH (s:Speaker {speaker_id: $speaker_id})
        RETURN s.speaker_id AS speaker_id,
               s.name AS name,
               s.embedding AS embedding,
               s.voice_preferences AS voice_preferences,
               s.created_at AS created_at,
               s.last_seen AS last_seen,
               s.sample_count AS sample_count
        """
        
        params = {'speaker_id': speaker_id}
        
        try:
            results = self.client.execute_query(query, params)
            if results:
                speaker = results[0]
                # Convert embedding back to numpy array
                speaker['embedding'] = np.array(speaker['embedding'])
                # Parse voice preferences
                if speaker.get('voice_preferences'):
                    speaker['voice_preferences'] = json.loads(speaker['voice_preferences'])
                else:
                    speaker['voice_preferences'] = {'voice_name': 'default', 'speed': 1.0, 'pitch': 0.0}
                return speaker
            return None
        except Exception as e:
            logger.error(f"Failed to get speaker: {e}")
            return None
    
    def get_all_speakers(self) -> List[Dict]:
        """
        Get all speaker profiles.
        
        Returns:
            List of speaker data (without embeddings for efficiency)
        """
        query = """
        MATCH (s:Speaker)
        RETURN s.speaker_id AS speaker_id,
               s.name AS name,
               s.created_at AS created_at,
               s.last_seen AS last_seen,
               s.sample_count AS sample_count
        ORDER BY s.last_seen DESC
        """
        
        try:
            results = self.client.execute_query(query)
            return results
        except Exception as e:
            logger.error(f"Failed to get speakers: {e}")
            return []
    
    def update_speaker(
        self,
        speaker_id: str,
        name: Optional[str] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Update speaker profile.
        
        Args:
            speaker_id: Speaker identifier
            name: New name (optional)
            embedding: New embedding (optional)
            
        Returns:
            True if successful, False otherwise
        """
        updates = []
        params = {'speaker_id': speaker_id}
        
        if name is not None:
            updates.append("s.name = $name")
            params['name'] = name
        
        if embedding is not None:
            updates.append("s.embedding = $embedding")
            updates.append("s.sample_count = s.sample_count + 1")
            params['embedding'] = embedding.tolist()
        
        if not updates:
            return False
        
        updates.append("s.last_seen = datetime()")
        
        query = f"""
        MATCH (s:Speaker {{speaker_id: $speaker_id}})
        SET {', '.join(updates)}
        RETURN s
        """
        
        try:
            result = self.client.execute_query(query, params)
            logger.info(f"Updated speaker: {speaker_id}")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Failed to update speaker: {e}")
            return False
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """
        Delete speaker profile.
        
        Args:
            speaker_id: Speaker identifier
            
        Returns:
            True if successful, False otherwise
        """
        query = """
        MATCH (s:Speaker {speaker_id: $speaker_id})
        DETACH DELETE s
        """
        
        params = {'speaker_id': speaker_id}
        
        try:
            self.client.execute_query(query, params)
            logger.info(f"Deleted speaker: {speaker_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete speaker: {e}")
            return False
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get all speaker embeddings for identification.
        
        Returns:
            Dictionary mapping speaker_id to embedding
        """
        query = """
        MATCH (s:Speaker)
        RETURN s.speaker_id AS speaker_id,
               s.embedding AS embedding
        """
        
        try:
            results = self.client.execute_query(query)
            embeddings = {}
            for result in results:
                speaker_id = result['speaker_id']
                embedding = np.array(result['embedding'])
                embeddings[speaker_id] = embedding
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return {}
    
    def get_voice_preferences(self, speaker_id: str) -> Dict:
        """
        Get TTS voice preferences for a speaker.
        
        Args:
            speaker_id: Speaker identifier
            
        Returns:
            Voice preferences dictionary or default preferences
        """
        query = """
        MATCH (s:Speaker {speaker_id: $speaker_id})
        RETURN s.voice_preferences AS voice_preferences
        """
        
        params = {'speaker_id': speaker_id}
        
        try:
            results = self.client.execute_query(query, params)
            if results and results[0].get('voice_preferences'):
                return json.loads(results[0]['voice_preferences'])
            else:
                # Return default preferences
                return {
                    'voice_name': 'default',
                    'speed': 1.0,
                    'pitch': 0.0
                }
        except Exception as e:
            logger.error(f"Failed to get voice preferences: {e}")
            return {'voice_name': 'default', 'speed': 1.0, 'pitch': 0.0}
    
    def update_voice_preferences(self, speaker_id: str, voice_preferences: Dict) -> bool:
        """
        Update TTS voice preferences for a speaker.
        
        Args:
            speaker_id: Speaker identifier
            voice_preferences: Voice preferences dictionary:
                {
                    'voice_name': str,  # TTS voice ID
                    'speed': float,     # Speaking rate (0.5-2.0)
                    'pitch': float      # Pitch shift (-10 to +10)
                }
            
        Returns:
            True if successful, False otherwise
        """
        query = """
        MATCH (s:Speaker {speaker_id: $speaker_id})
        SET s.voice_preferences = $voice_preferences,
            s.last_seen = datetime()
        RETURN s
        """
        
        params = {
            'speaker_id': speaker_id,
            'voice_preferences': json.dumps(voice_preferences)
        }
        
        try:
            result = self.client.execute_query(query, params)
            logger.info(f"Updated voice preferences for {speaker_id}: {voice_preferences}")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Failed to update voice preferences: {e}")
            return False


def main():
    """Test the profile manager."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    import config
    from src.database.neo4j_client import Neo4jClient
    
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    manager = SpeakerProfileManager(client)
    
    print("Testing Speaker Profile Manager\n")
    
    # Test create
    test_embedding = np.random.randn(192)
    speaker_data = manager.create_speaker(
        speaker_id="test_001",
        name="Test Speaker",
        embedding=test_embedding
    )
    print(f"Created: {speaker_data}")
    
    # Test get
    speaker = manager.get_speaker("test_001")
    print(f"\nRetrieved: {speaker['name']}, embedding shape: {speaker['embedding'].shape}")
    
    # Test get all
    all_speakers = manager.get_all_speakers()
    print(f"\nAll speakers: {len(all_speakers)}")
    
    # Test delete
    manager.delete_speaker("test_001")
    print("\nDeleted test speaker")
    
    client.close()


if __name__ == '__main__':
    main()
