// Freepik API Integration
const FREEPIK_API_KEY = 'FPSX91458a8c93d4f38269317273e044d399';
const FREEPIK_BASE_URL = 'https://api.freepik.com/v1';

const headers: Record<string, string> = {
  'x-freepik-api-key': FREEPIK_API_KEY,
  'Content-Type': 'application/json'
};

// Icon Search
export interface FreepikIcon {
  id: string;
  name: string;
  description: string;
  thumbnail: string;
  downloadUrl: string;
}

export async function searchIcons(query: string, limit = 10): Promise<FreepikIcon[]> {
  try {
    const response = await fetch(
      `${FREEPIK_BASE_URL}/icons?query=${encodeURIComponent(query)}&limit=${limit}`,
      { headers }
    );

    if (!response.ok) {
      throw new Error(`Freepik API error: ${response.status}`);
    }

    const data = await response.json();
    return data.data || [];
  } catch (error) {
    console.error('Freepik icon search error:', error);
    return [];
  }
}

// Resource Search (Photos, Vectors, PSDs)
export interface FreepikResource {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  type: 'photo' | 'vector' | 'psd' | 'ai';
  premium: boolean;
  downloadUrl: string;
}

export async function searchResources(
  query: string,
  type?: 'photo' | 'vector' | 'psd',
  limit = 10
): Promise<FreepikResource[]> {
  try {
    let url = `${FREEPIK_BASE_URL}/resources?query=${encodeURIComponent(query)}&limit=${limit}`;
    if (type) {
      url += `&type=${type}`;
    }

    const response = await fetch(url, { headers });

    if (!response.ok) {
      throw new Error(`Freepik API error: ${response.status}`);
    }

    const data = await response.json();
    return data.data || [];
  } catch (error) {
    console.error('Freepik resource search error:', error);
    return [];
  }
}

// AI Image Generation
export interface GenerationTask {
  taskId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  imageUrl?: string;
}

export async function generateImage(
  prompt: string,
  options: {
    model?: 'mystic' | 'flux' | 'seedream';
    resolution?: '1k' | '2k' | '4k';
  } = {}
): Promise<GenerationTask> {
  try {
    const response = await fetch(`${FREEPIK_BASE_URL}/ai/text-to-image`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        prompt,
        model: options.model || 'mystic',
        resolution: options.resolution || '2k'
      })
    });

    if (!response.ok) {
      throw new Error(`Freepik AI error: ${response.status}`);
    }

    const data = await response.json();
    return {
      taskId: data.task_id,
      status: 'pending'
    };
  } catch (error) {
    console.error('Freepik AI generation error:', error);
    return {
      taskId: '',
      status: 'failed'
    };
  }
}

// Check generation status
export async function checkGenerationStatus(taskId: string): Promise<GenerationTask> {
  try {
    const response = await fetch(`${FREEPIK_BASE_URL}/ai/tasks/${taskId}`, { headers });

    if (!response.ok) {
      throw new Error(`Freepik task check error: ${response.status}`);
    }

    const data = await response.json();
    return {
      taskId,
      status: data.status,
      imageUrl: data.image_url
    };
  } catch (error) {
    console.error('Freepik task check error:', error);
    return {
      taskId,
      status: 'failed'
    };
  }
}

// Download icon
export async function downloadIcon(iconId: string, format: 'svg' | 'png' = 'svg'): Promise<string | null> {
  try {
    const response = await fetch(
      `${FREEPIK_BASE_URL}/icons/${iconId}/download?format=${format}`,
      { headers }
    );

    if (!response.ok) {
      throw new Error(`Freepik download error: ${response.status}`);
    }

    const data = await response.json();
    return data.url || null;
  } catch (error) {
    console.error('Freepik download error:', error);
    return null;
  }
}

// Download resource
export async function downloadResource(resourceId: string): Promise<string | null> {
  try {
    const response = await fetch(
      `${FREEPIK_BASE_URL}/resources/${resourceId}/download`,
      { headers }
    );

    if (!response.ok) {
      throw new Error(`Freepik download error: ${response.status}`);
    }

    const data = await response.json();
    return data.url || null;
  } catch (error) {
    console.error('Freepik download error:', error);
    return null;
  }
}
