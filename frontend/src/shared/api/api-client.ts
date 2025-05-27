export type RequestType = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

export interface ApiClientConfig {
    baseURL?: string;
    headers?: Record<string, string>;
}

const createApiClient = <TResponse = unknown>(config: ApiClientConfig = {}) => {
    const baseURL = import.meta.env.VITE_BASE_API_URL;
    const headers = {
        'Content-Type': 'application/json',
        ...config.headers,
    };

    const request = async <T = TResponse>(
        method: RequestType,
        url: string,
        data?: unknown,
        config?: RequestInit,
    ): Promise<T> => {
        const fullUrl = `${baseURL}${url}`;
        const requestConfig: RequestInit = {
            method,
            headers: {
                ...headers,
                ...config?.headers,
            },
            ...config,
        };

        if (data && method !== 'GET') {
            requestConfig.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(fullUrl, requestConfig);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || response.statusText);
            }
            return response.json();
        } catch (error) {
            if (error instanceof Error) {
                throw new Error(error.message);
            }
            throw error;
        }
    };

    return {
        get: <T = TResponse>(url: string, config?: RequestInit) => request<T>('GET', url, undefined, config),
        post: <T = TResponse>(url: string, data?: unknown, config?: RequestInit) => request<T>('POST', url, data, config),
        put: <T = TResponse>(url: string, data?: unknown, config?: RequestInit) => request<T>('PUT', url, data, config),
        delete: <T = TResponse>(url: string, config?: RequestInit) => request<T>('DELETE', url, undefined, config),
        patch: <T = TResponse>(url: string, data?: unknown, config?: RequestInit) => request<T>('PATCH', url, data, config),
    };
};

// Create a default instance
export const apiClient = createApiClient();

// Export the factory function for custom instances
export default createApiClient; 