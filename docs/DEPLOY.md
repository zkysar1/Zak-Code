# Keeping `zakcode webapp` running

The CLI cockpit relaunches itself on Enter because its view and its engine share one
process. In the webapp the engine **is** the server, so the one supervision job left is
keeping `zakcode webapp` alive on the box. A server launched by hand in a terminal is the
fragile spot: it dies with the terminal, with the SSH session, with the first crash, and
nobody notices until the mind stops answering. This page is the one supported answer —
one unit shape, one restart policy, no knobs.

## Linux box with systemd (the normal case)

The unit ships in the repo at [`deploy/systemd/zakcode-webapp.service`](../deploy/systemd/zakcode-webapp.service).
Its header repeats these steps. As root:

```bash
# 1. A service account and the workspace it serves (identity, rules, skills, sessions).
useradd --system --create-home --shell /usr/sbin/nologin zakcode
install -d -o zakcode -g zakcode /srv/zakcode/workspace

# 2. zakcode for that account — the command lands in ~zakcode/.local/bin. A box that runs a
#    Vertex AI model installs "zakcode[server,google] @ …" instead.
sudo -u zakcode uv tool install "zakcode[server] @ git+https://github.com/zkysar1/Zak-Code.git"

# 3. Keys and settings, readable by the service account only.
install -o root -g zakcode -m 0640 /dev/null /etc/zakcode/.env
$EDITOR /etc/zakcode/.env      # ZAKCODE_DEFAULT_MODEL=..., the provider key, ZAKCODE_AUTH_TOKEN=... if exposed

# 4. The unit.
cp deploy/systemd/zakcode-webapp.service /etc/systemd/system/
systemd-analyze verify /etc/systemd/system/zakcode-webapp.service
systemctl daemon-reload && systemctl enable --now zakcode-webapp

# 5. It answers.
curl -s http://127.0.0.1:8000/health
```

What the unit fixes, and why:

| Line | Why |
| --- | --- |
| `Restart=always`, `RestartSec=3` | Every exit comes back — a crash, an OOM kill, a `zakcode update` that replaced the install — without a human relaunching it. |
| `StartLimitIntervalSec=0` | systemd's default gives up after 5 starts in 10 s. A bad key or a provider outage would otherwise leave the box without its mind until someone noticed. |
| `EnvironmentFile=-/etc/zakcode/.env` | The box-level config home. The `-` makes a missing file a no-op rather than a start failure; the per-user `~/.zakcode/.env` layering ([CONFIG.md](CONFIG.md)) still applies underneath. |
| `--host 127.0.0.1` | Loopback-only. To reach it from elsewhere put a reverse proxy in front, or set `ZAKCODE_AUTH_TOKEN` in the env file and bind `0.0.0.0` — the server refuses a non-loopback bind without a token unless you pass `--insecure`. |
| `--workspace /srv/zakcode/workspace` | The mind the server serves: one workspace per served mind, its sessions under `<workspace>/.zakcode/sessions/` (ADR-0032). |
| `--no-new-session` | A restart goes on with the chat the pages are on. Without it every start opens a new chat, and every open page follows the restarted server into it, in the middle of a reply. Keep it in any unit, script or container command you write yourself. |
| `NoNewPrivileges`, `PrivateTmp` | Hardening that stays out of the mind's way — it only needs its workspace and its config. |

Only the three lines marked `box-specific` in the unit change between boxes: the
workspace path, the `PATH` that holds the `zakcode` command for the service account, and
the bind. Everything else is the policy.

Day two:

```bash
journalctl -u zakcode-webapp -f                            # logs
sudo -u zakcode zakcode update                             # a new build
sudo -u zakcode zakcode update --extra google              # a new build, with Vertex AI's libraries
systemctl restart zakcode-webapp                           # after changing /etc/zakcode/.env
systemctl status zakcode-webapp                            # is it up, when did it last restart
```

`zakcode update` reinstalls in place from the install's own recorded source (a git URL or a
local checkout) and prints old → new build identity. `--extra NAME` adds one of zakcode's
optional extras on the way, and the updates after it keep it (ADR-0340). The running server notices the new build
within five seconds and, once no turn is in flight, restarts itself into it on the same chat:
it finishes the turn it is running, takes no new one on the old build, and replaces its own
process, so the pid and the unit stay as they were and the journal says
`restarted into build …` (ADR-0333). A page left open reconnects. A server older than that
change does not look for new builds: restart the unit once to bring it onto a build that does.
The new build reads the workspace `.env` and `~/.zakcode/.env` afresh, but it keeps the unit's
own environment, so a change to `/etc/zakcode/.env` (read by `EnvironmentFile=`) or to the unit
still needs `systemctl restart zakcode-webapp`. A run a host manages (any `ZAKCODE_RUN_*`
ending set) keeps its build until the host restarts it.

A local-checkout install is reinstalled with every dependency at the version the checkout's
`uv.lock` pins, and an editable install stays editable (ADR-0291). On Windows the reinstall
finishes in a second process after the command exits, because Windows will not replace the
files of a program that is running: the command refuses while another program runs from the
install (it names the pid), and the second process puts the old install back if anything
fails. If the programs cannot be listed the update goes on, and a program still running from the old
install keeps its files from being deleted: the log says so and leaves the folder, which is safe to delete by
hand once that program has exited, and the next update removes it. Restart that program to use the new
install. Its progress and result are in `~/.zakcode/update/update.log`, which ends
`update complete` or `update failed`; before it moves the old install the log says how to put
it back by hand, for the case that the process is killed. An update that stops leaves the
checkout pulled, and says so with the `git reset --hard` that returns it.

The build backend is pinned as well: `pyproject.toml` names one exact hatchling release, and
every build of the project uses it, an update's included (ADR-0300). hatchling's own
dependencies are not pinned. These pins fix versions, not bytes: `uv tool install` does not
check a hash in a constraints file, so the constraints an update exports from `uv.lock` carry
none.

The reinstall runs in `~/.zakcode/update` and names its constraints file there without the
folder, because uv before 0.13.0 splits a constraints path at a space and a profile folder can
hold one (ADR-0301). An install built between ADR-0291 and this change passes the full path, so
on such a uv its next update stops with `File not found` and keeps the old install: upgrade uv
to 0.13.0 or later first (`uv self update` where uv came from its own installer).

## Hand-managed hosts

A host nobody reaches by fleet automation gets the same unit, installed by hand with the
steps above — the deliverable there is the documented unit, not a script that logs in. Do
not run the server from a terminal "for now"; that is the exact failure this page exists
to end.

## Containers

The same policy, spelled in Compose. There is no image in this repo; build one from the
package (`pip install "zakcode[server]"`) and keep the workspace and the env file outside the
container so they outlive it:

```yaml
services:
  zakcode:
    image: your-registry/zakcode:latest
    command: ["zakcode", "webapp", "--host", "0.0.0.0", "--port", "8000", "--workspace", "/workspace", "--no-new-session"]
    env_file:
      - /etc/zakcode/.env            # must set ZAKCODE_AUTH_TOKEN: 0.0.0.0 inside the container is non-loopback
    volumes:
      - /srv/zakcode/workspace:/workspace
    ports:
      - "127.0.0.1:8000:8000"        # publish on loopback; a reverse proxy takes it from there
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

`restart: always` is the container form of the unit's policy; the health check gives the
orchestrator the same `/health` the unit's install step curls.

## Behind a reverse proxy

Send every path to the server, the page at `/` included. A proxy that answers `/` with a copy
of the page saved on its own disk keeps the page on the build the copy was saved from, while
each `zakcode update` moves the server on: the browser then runs an old page against a newer
server, and no update reaches it. That is how chat switching stayed broken on one box after its
fix had shipped (ADR-0342). The server sends the page with `Cache-Control: no-cache`, so a
browser asks for it again each time it opens it.

An open page reloads itself when the server it watches serves another page, as after a restart
into a new build (ADR-0345). Each watch response names the server's page in its `X-Page-Digest`
header, which the proxy passes back as it passes back `X-Bus-Epoch`. A proxy that drops that
header leaves every open page on its old script, and nothing on the page says so, as with a
server too old to send it. The reload asks for `/?page=<that name>`, and the page keeps the name
in the browser's session storage as well, so a proxy that drops the query from the address loses
nothing. A page that reloads and is given the same old page again does not reload again: a note
over the composer says that something between the browser and the server answers with a saved
copy of the page, which is the problem this section describes.

A server with a token (`ZAKCODE_AUTH_TOKEN`) asks for it on every path but `/health`, the page
included, and a browser has no way to send it: the proxy adds `Authorization: Bearer <token>` to
each request it passes on, and does its own sign-in in front.

To check that a browser gets the server's own page, compare the two. Give the first `curl` the
proxy's sign-in if it has one, and the second the token if the server has one
(`-H "Authorization: Bearer $ZAKCODE_AUTH_TOKEN"`):

```bash
diff <(curl -s https://your.host/) <(curl -s http://127.0.0.1:8000/) >/dev/null && echo "same page" || echo "DIFFERENT"
```

## Checking a box

```bash
systemctl is-enabled zakcode-webapp && systemctl is-active zakcode-webapp
systemd-analyze verify /etc/systemd/system/zakcode-webapp.service
curl -s http://127.0.0.1:8000/health
```

Three green answers mean the mind comes back on its own. Anything else is a hand-launched
server waiting to die.
