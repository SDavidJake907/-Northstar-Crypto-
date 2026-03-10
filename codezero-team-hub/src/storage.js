// Local storage wrapper that mimics the Claude artifact storage API
// Data persists in localStorage across browser sessions

const storage = {
  async get(key) {
    try {
      const val = localStorage.getItem(`cz_${key}`);
      if (val === null) throw new Error('Key not found');
      return { key, value: val, shared: false };
    } catch (e) {
      throw e;
    }
  },

  async set(key, value) {
    try {
      localStorage.setItem(`cz_${key}`, value);
      return { key, value, shared: false };
    } catch (e) {
      console.error('Storage set error:', e);
      return null;
    }
  },

  async delete(key) {
    try {
      localStorage.removeItem(`cz_${key}`);
      return { key, deleted: true, shared: false };
    } catch (e) {
      return null;
    }
  },

  async list(prefix = '') {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k.startsWith(`cz_${prefix}`)) {
        keys.push(k.replace('cz_', ''));
      }
    }
    return { keys, prefix, shared: false };
  }
};

// Mount to window so the app code works unchanged
window.storage = storage;

export default storage;
