# COMPONENT 3: TTS + LIP-SYNC IMPLEMENTATION CHECKLIST

## QUICK START GUIDE (Week 1 Implementation)

### Phase 1: API Setup (Days 1-2)

#### ElevenLabs Setup
- [ ] Create account on elevenlabs.io
- [ ] Generate API key
- [ ] Test REST endpoint with sample text
- [ ] Set up WebSocket streaming connection
  ```bash
  # Test latency
  curl -X POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream \
    -H "xi-api-key: your-key" \
    -d '{"text":"Hello world"}' > test.mp3
  ```
- [ ] Select 3-5 voice IDs for game characters
- [ ] Configure voice settings (stability, similarity_boost)
- [ ] Set up payment method ($5/mo minimum)

#### Alternative: Coqui XTTS v2 (if self-hosting)
- [ ] Choose cloud provider (Runpod.io recommended for ease)
- [ ] Deploy Docker container with Coqui TTS
- [ ] Test endpoint latency from your region
- [ ] Set up FastAPI wrapper for consistency
- [ ] Test voice cloning with 6-second sample

### Phase 2: Game Engine Integration (Days 3-5)

#### For Unity
- [ ] Create C# script: `TTSManager.cs`
- [ ] Implement WebRequest coroutine for ElevenLabs API
- [ ] Add streaming audio support (AudioSource + streamed clip)
- [ ] Test latency: measure time from "play" to "first audio"
- [ ] Create fallback cache for common phrases
- [ ] Test with game dialogue system

#### For Unreal Engine 5
- [ ] Create C++ module: `FTTSManager`
- [ ] Implement HTTP requests via FHttpModule
- [ ] Connect AudioComponent to TTS output
- [ ] Set up MetaHuman audio sync
- [ ] Test with Blueprint dialogue system

### Phase 3: Lip-Sync Integration (Days 6-7)

#### Wav2Lip Setup
- [ ] Clone Wav2Lip GitHub repo
- [ ] Download pre-trained model (download link in repo)
- [ ] Set up Python environment with dependencies
  ```bash
  pip install torch torchvision librosa imageio imageio-ffmpeg
  git clone https://github.com/justinzhao/Wav2Lip_288.git
  cd Wav2Lip_288
  # Download checkpoint-step-000210.pth
  ```
- [ ] Test with sample video + audio
- [ ] Measure inference latency on your hardware
- [ ] Create wrapper script for batch processing

#### UE5 MetaHuman Integration (Alternative)
- [ ] Import MetaHuman character into project
- [ ] Set up audio playback component
- [ ] Enable audio-driven lip-sync in Animation Blueprint
- [ ] Test synchronization quality

### Phase 4: Testing & Optimization (Week 2)

- [ ] Measure end-to-end latency: text input → audio output → lip-sync
- [ ] Record baseline: current latency vs target (<200ms)
- [ ] Optimize network calls (parallel requests for multiple NPCs)
- [ ] Cache top 50 dialogue phrases locally
- [ ] Test fallback when API unavailable
- [ ] Verify voice quality (subjective MOS test with team)
- [ ] Profile GPU/CPU usage for lip-sync generation

---

## PRODUCTION DEPLOYMENT CHECKLIST

### Pre-Launch (2 weeks before)

#### Infrastructure
- [ ] Set up error logging (Sentry, DataDog)
- [ ] Configure backup TTS provider (OpenAI fallback)
- [ ] Set up cache layer (Redis for common phrases)
- [ ] Load test: simulate expected concurrent dialogue requests
- [ ] Set up monitoring dashboard for API latency/errors
- [ ] Configure auto-scaling for backend

#### Quality Assurance
- [ ] Voice acting QA: 3+ native speakers per language
- [ ] Lip-sync accuracy testing on target avatars
- [ ] Latency measurements in production environment
- [ ] Emotion/expression verification (if using SadTalker)
- [ ] Edge case testing (special characters, emojis, RTL text)

#### Security
- [ ] Secure API keys (use environment variables, not hardcoded)
- [ ] Rate limiting: set per-user limits
- [ ] Input validation: sanitize all user text before TTS
- [ ] GDPR compliance: log retention policy
- [ ] Monitor for abuse: unusually high API calls

### Launch Day
- [ ] Verify all systems are responding
- [ ] Monitor error rates (should be <0.1%)
- [ ] Check latency metrics in production
- [ ] Have fallback manual dialogue system ready
- [ ] Team on standby for first-hour issues

### Post-Launch (Week 1)
- [ ] Monitor user feedback for audio quality
- [ ] Adjust voice settings based on feedback
- [ ] Optimize cache based on actual dialogue patterns
- [ ] Fine-tune latency vs quality trade-offs

---

## ARCHITECTURE TEMPLATES

### Unity Implementation (Basic)

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;

public class GameAIVoice : MonoBehaviour
{
    [SerializeField] private string elevenLabsAPIKey = "your-key-here";
    [SerializeField] private string voiceID = "21m00Tcm4TlvDq8ikWAM"; // Rachel voice
    
    private AudioSource audioSource;
    private Dictionary<string, AudioClip> dialogueCache = new Dictionary<string, AudioClip>();
    private string apiURL = "https://api.elevenlabs.io/v1/text-to-speech";

    void Start()
    {
        audioSource = GetComponent<AudioSource>();
        
        // Pre-cache common phrases
        CacheCommonPhrases();
    }

    public void PlayDialogue(string dialogueText)
    {
        // Check cache first
        if (dialogueCache.ContainsKey(dialogueText))
        {
            audioSource.PlayOneShot(dialogueCache[dialogueText]);
            return;
        }

        // Request from API
        StartCoroutine(RequestTTS(dialogueText));
    }

    IEnumerator RequestTTS(string text)
    {
        string url = $"{apiURL}/{voiceID}";
        
        var body = new { 
            text = text,
            model_id = "eleven_monolingual_v1",
            voice_settings = new { 
                stability = 0.5f, 
                similarity_boost = 0.75f 
            }
        };

        using (UnityWebRequest www = new UnityWebRequest(url, "POST"))
        {
            byte[] jsonToSend = System.Text.Encoding.UTF8.GetBytes(JsonUtility.ToJson(body));
            www.uploadHandler = new UploadHandlerRaw(jsonToSend);
            www.downloadHandler = new DownloadHandlerAudioClip("", AudioType.MPEG);
            www.SetRequestHeader("Content-Type", "application/json");
            www.SetRequestHeader("xi-api-key", elevenLabsAPIKey);

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
                dialogueCache[text] = clip; // Cache for future use
                audioSource.PlayOneShot(clip);
            }
            else
            {
                Debug.LogError("TTS Error: " + www.error);
            }
        }
    }

    void CacheCommonPhrases()
    {
        string[] commonPhrases = {
            "Hello, adventurer!",
            "Welcome to my inn.",
            "Quest accepted!",
            "Thank you for your help.",
            "See you next time!"
        };

        foreach (string phrase in commonPhrases)
        {
            StartCoroutine(RequestTTS(phrase));
        }
    }
}
```

### Unreal Engine 5 Implementation (C++)

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "Http.h"
#include "Sound/SoundBase.h"
#include "GameAICharacter.generated.h"

UCLASS()
class YOURGAME_API AGameAICharacter : public ACharacter
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Voice")
    FString ElevenLabsAPIKey;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Voice")
    FString VoiceID;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Audio")
    class UAudioComponent* AudioComponent;

private:
    FHttpModule* HttpModule;
    TMap<FString, USoundWave*> DialogueCache;

public:
    AGameAICharacter();
    
    virtual void BeginPlay() override;
    
    UFUNCTION(BlueprintCallable, Category = "Voice")
    void PlayDialogue(const FString& DialogueText);

private:
    void RequestTTSAudio(const FString& Text);
    void OnTTSResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful);
    void CacheCommonPhrases();
};
```

### Python Backend (FastAPI Wrapper)

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from elevenlabs import Client, VoiceSettings
from TTS.api import TTS
import os

app = FastAPI()

# Initialize clients
elevenlabs_client = Client(api_key=os.getenv("ELEVENLABS_API_KEY"))
coqui_tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

CACHE = {}

@app.post("/tts/elevenlabs/stream")
async def tts_elevenlabs_stream(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
    """Stream TTS from ElevenLabs with minimal latency"""
    try:
        audio = elevenlabs_client.generate(
            text=text,
            voice=voice_id,
            model="eleven_monolingual_v1",
            stream=True
        )
        
        return StreamingResponse(audio, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/coqui/stream")
async def tts_coqui_stream(text: str, language: str = "en", speaker_wav: str = None):
    """Stream TTS from Coqui XTTS v2 (open-source alternative)"""
    try:
        # Generate audio synchronously (Coqui doesn't support streaming yet)
        wav = coqui_tts.tts(
            text=text,
            language=language,
            speaker_wav=speaker_wav
        )
        
        # Convert to bytes and stream
        import io
        from scipy.io import wavfile
        
        audio_bytes = io.BytesIO()
        wavfile.write(audio_bytes, rate=24000, data=wav)
        audio_bytes.seek(0)
        
        return StreamingResponse(audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lipsync/wav2lip")
async def generate_lip_sync(audio_file: str, video_file: str):
    """Generate lip-sync using Wav2Lip model"""
    try:
        import subprocess
        import json
        
        # Call Wav2Lip inference
        result = subprocess.run([
            "python", "inference.py",
            "--checkpoint_path", "checkpoints/wav2lip.pth",
            "--face", video_file,
            "--audio", audio_file,
            "--outfile", "output_video.mp4",
            "--face_det_checkpoint", "face_detection/detection_YOLO_nano.pt"
        ], capture_output=True)
        
        if result.returncode == 0:
            return {"status": "success", "output_path": "output_video.mp4"}
        else:
            raise Exception(result.stderr.decode())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "cache_size": len(CACHE)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## PERFORMANCE OPTIMIZATION TIPS

### Latency Optimization
1. **Parallel requests:** Send dialogue for multiple NPCs simultaneously
2. **Streaming chunks:** Start playback before full audio generated
3. **Edge caching:** Cache outputs in CDN for common phrases
4. **Hardware acceleration:** Use GPU for lip-sync generation
5. **Connection pooling:** Reuse HTTP connections for multiple requests

### Cost Optimization
1. **Phrase caching:** Pre-generate 50-100 common dialogue lines
2. **Multi-provider:** Use Speechmatics for common phrases, ElevenLabs for important ones
3. **Batch generation:** Generate all dialogue during off-hours
4. **Rate limiting:** Prevent abuse/duplicate requests
5. **Character filters:** Only TTS dialogue that's actually heard by player

### Quality Optimization
1. **Voice settings tuning:** Adjust stability/similarity for character
2. **Audio normalization:** Consistent volume across all dialogue
3. **Post-processing:** Add subtle reverb/EQ for game world immersion
4. **Language-specific:** Different voice settings per language
5. **A/B testing:** Compare voices with players for preference

---

## TESTING CHECKLIST

### Unit Tests
- [ ] API response parsing
- [ ] Cache hit/miss logic
- [ ] Error handling (network failure, API error)
- [ ] Voice setting validation

### Integration Tests
- [ ] Game engine → TTS API full flow
- [ ] Fallback provider switching
- [ ] Lip-sync with game avatar animation
- [ ] Multiple concurrent dialogue requests

### Performance Tests
- [ ] Latency: measure TTFB + total generation time
- [ ] Throughput: concurrent requests handling
- [ ] Memory: cache memory usage
- [ ] GPU: VRAM usage for lip-sync

### Quality Tests
- [ ] Subjective voice naturalness (MOS score)
- [ ] Lip-sync accuracy on video
- [ ] Emotional expression if applicable
- [ ] Language pronunciation

---

## ESTIMATED COSTS & TIMELINES

### Small Game (10K DAU)
- Development: 40 hours ($2,000-4,000)
- Infrastructure: $180-200/month
- Voice talent: $500-1,000
- **Total first month:** $2,700-5,200

### Medium Game (100K DAU)
- Development: 100 hours ($5,000-10,000)
- Infrastructure: $600-800/month
- Voice talent: $2,000-5,000
- **Total first month:** $7,600-15,800

### Large Game (1M+ DAU)
- Development: 300+ hours ($15,000-30,000)
- Infrastructure: $30,000-50,000/month
- Voice talent: $10,000-50,000
- **Total first month:** $55,000-130,000

---

## COMMON ISSUES & SOLUTIONS

### Issue: High Latency (>300ms)
**Solutions:**
- Switch to ElevenLabs Flash 2.5 if using slower provider
- Enable local caching for common phrases
- Reduce distance to API server (use regional endpoint)
- Pre-generate dialogue during loading screens

### Issue: Poor Lip-Sync Quality
**Solutions:**
- Increase Wav2Lip model accuracy (higher res input)
- Use SadTalker for more natural head movement
- Ensure audio quality is good (no noise)
- Test on target avatar hardware

### Issue: Audio Quality Issues
**Solutions:**
- Adjust voice settings (stability, similarity)
- Normalize audio output
- Test different voice IDs
- Check audio format/bitrate

### Issue: High API Costs
**Solutions:**
- Increase cache hit rate (pre-generate more phrases)
- Use cheaper provider for non-critical dialogue
- Implement rate limiting
- Batch generate offline

---

## RECOMMENDED NEXT STEPS

1. **This Week:** Complete Phase 1 (API setup) + Phase 2 (game integration)
2. **Next Week:** Complete Phase 3 (lip-sync) + Phase 4 (optimization)
3. **Week 3:** Collect voice feedback, fine-tune settings
4. **Week 4:** Load testing, prepare for production
5. **Week 5:** Soft launch, monitor real-world performance

---

**Document Updated:** December 4, 2025  
**Status:** Ready for implementation  
**Support Docs:** See full research document (tts-lipsync-research.md) for detailed info