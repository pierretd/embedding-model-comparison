from daytona_sdk import Daytona, DaytonaConfig

# Define the configuration
config = DaytonaConfig(
  api_key="dtn_ad3e1a6a39ebe3a24061ab95112fbd487051d588443e857e5cc8798fc6d1d3c3",
  server_url="https://app.daytona.io/api",
  target="us"
)

# Initialize the Daytona client
daytona = Daytona(config)

# Create the Sandbox instance
sandbox = daytona.create()

# Run the code securely inside the Sandbox
response = sandbox.process.code_run('print("Hello World from code!")')
if response.exit_code != 0:
  print(f"Error: {response.exit_code} {response.result}")
else:
    print(response.result)

sandbox.fs.create_folder(f"{sandbox.get_user_root_dir()}/workspace/data", mode="755")

daytona.remove(sandbox)