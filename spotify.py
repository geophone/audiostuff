import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp=spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="30a1e64dd1a342b38b0759a1deda5058", client_secret='3485feb2673442879bc6217871b2bd13',redirect_uri="http://127.0.0.1:3000/",scope="user-library-read"))

print(sp.playlist('2EukjTGPrm0ADFgRKFXxlr'))



