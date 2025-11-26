import axios from 'axios';

// Separate axios instance for Flask server (no credentials needed)
const flaskAxios = axios.create({
  withCredentials: false, // Flask server doesn't require authentication
});

export default flaskAxios;

