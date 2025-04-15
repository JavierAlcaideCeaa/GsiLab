import praw
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

class RedditAPI:
    def __init__(self):
        self.client_id = "MllQGKXnomD7tpJfVbR8pg"
        self.client_secret = "Ut-IcyKDQYi5GVpLjiIigMjN7wz7Rg"
        self.user_agent = "Feeling-Language-998"
        print(f"Client ID: {self.client_id}")
        print(f"Client Secret: {self.client_secret}")
        print(f"User Agent: {self.user_agent}")

        # Verificar que las credenciales se cargaron correctamente
        if not all([self.client_id, self.client_secret, self.user_agent]):
            raise ValueError("Faltan credenciales de Reddit en el archivo .env")

        # Inicializar el cliente de Reddit
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

    def get_posts(self, subreddit, limit=10):
        # Obtener publicaciones de un subreddit
        posts = []
        for post in self.reddit.subreddit(subreddit).hot(limit=limit):
            posts.append({
                'title': post.title,
                'score': post.score,
                'url': post.url,
                'created_at': post.created_utc
            })
        return posts
