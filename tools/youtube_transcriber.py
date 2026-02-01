"""
OpenClaw - YouTube Transcriber Tool
===================================
Tool for fetching transcripts from YouTube videos.
"""

import json
import logging
from typing import Any, Optional
import httpx
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

class YoutubeTranscriberTool:
    """
    Tool to extract transcripts from YouTube videos.
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def definition(self) -> dict:
        """Return the tool definition for LLM."""
        return {
            "name": "youtube_transcriber",
            "description": (
                "Extract transcript/captions from a YouTube video URL. "
                "Use this to get the text content of a video for analysis or summarization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "The full YouTube video URL (e.g., https://www.youtube.com/watch?v=...) or video ID."
                    },
                    "language": {
                        "type": "string",
                        "description": "Preferred language code (e.g., 'en', 'pt'). Default is 'en'.",
                        "default": "en"
                    }
                },
                "required": ["video_url"]
            }
        }
    
    async def execute(
        self,
        video_url: str,
        language: str = "en"
    ) -> dict[str, Any]:
        """
        Execute the transcript fetch.
        
        Args:
            video_url: URL or ID of the YouTube video
            language: Preferred language code
            
        Returns:
            Dictionary with transcript or error
        """
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            if not video_id:
                return {
                    "success": False,
                    "error": "Could not extract valid video ID from URL"
                }
            
            console.print(f"[blue]📺 Fetching transcript for video: {video_id}[/blue]")
            
            # Since we don't have youtube-transcript-api, we'll try a fallback or basic message
            # Ideally, we should check if youtube-transcript-api is installed, but for now 
            # we'll use a placeholder implementation or try to fetch if possible without complex deps.
            # Given the constraints and previous error, I will try to import youtube_transcript_api safely
            
            transcript_text = ""
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                
                # Fetch transcript
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language, 'en'])
                
                # specific formatter could be used, but simple join is enough
                transcript_text = "\n".join([item['text'] for item in transcript_list])
                
            except ImportError:
                return {
                    "success": False,
                    "error": "youtube_transcript_api library is not installed. Please install it to use this tool."
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to fetch transcript: {str(e)}"
                }
            
            if not transcript_text:
                return {
                    "success": False,
                    "error": "No transcript available for this video."
                }
                
            return {
                "success": True,
                "video_id": video_id,
                "transcript": transcript_text
            }
            
        except Exception as e:
            logger.error(f"YoutubeTranscriberTool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        if len(url) == 11 and " " not in url:
            return url
            
        import re
        # Standard https://www.youtube.com/watch?v=VIDEO_ID
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)
            
        # Short https://youtu.be/VIDEO_ID
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)
            
        return None
