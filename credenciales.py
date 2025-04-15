import praw

reddit = praw.Reddit(
    client_id="MllQGKXnomD7tpJfVbR8pg",
    client_secret="Ut-IcyKDQYi5GVpLjiIigMjN7wz7Rg",
    user_agent="Feeling-Language-998"
)

try:
    subreddit = reddit.subreddit("python")
    for post in subreddit.hot(limit=5):
        print(f"Title: {post.title}, Score: {post.score}")
except Exception as e:
    print(f"Error: {e}")