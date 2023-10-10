/**
 * JetBrains Space Automation
 * This Kotlin script file lets you automate build activities
 * For more info, see https://www.jetbrains.com/help/space/automation.html
 */

job("Prepare testcontainer image") {
    // do not run on git push
    startOn {
        gitPush { enabled = false }
    }

    kaniko {
        build {
            dockerfile = "./tests/testcontainer/Dockerfile"
            labels["vendor"] = "bastelquartier.de"
        }

        push("bastelquartier.registry.jetbrains.space/p/fapi-el/testcontainer/testcontainer") {
            tags {
                +"0.0.1"
            }
        }
    }
}

job("Run tests") {
    startOn {
        gitPush { enabled = false }
    }

    container(image = "bastelquartier.registry.jetbrains.space/p/fapi-el/testcontainer/testcontainer:0.0.1") {
        env["URL"] = "https://pypi.pkg.jetbrains.space/bastelquartier/p/fapi-el/controllogger/legacy"
        shellScript {
            content = """
                #echo Run tests...
                #pytest ./tests/
            """
        }
    }
}

job("Build and publish Package") {
    startOn {
        gitPush {
            anyBranchMatching {
                +"main"
            }
        }
    }
    container(image = "bastelquartier.registry.jetbrains.space/p/fapi-el/testcontainer/testcontainer:0.0.1") {
        env["pypi_token"] = "{{ project:pypi_token }}"
        shellScript {
            content = """
                echo Build package...
                python setup.py sdist -bV ${'$'}JB_SPACE_EXECUTION_NUMBER
                
                echo Publish package to space ...
                twine upload --repository-url https://pypi.pkg.jetbrains.space/bastelquartier/p/fapi-el/controllogger/legacy -u ${'$'}JB_SPACE_CLIENT_ID -p ${'$'}JB_SPACE_CLIENT_SECRET dist/*
                
                if [ ${'$'}pypi_token == "" ]; then
                    echo Publish package to pypi failed. No token found.
                    exit 1
                fi
                
                echo Publish package to pypi ...
                twine upload --repository-url https://upload.pypi.org/legacy/ -u __token__ -p ${'$'}pypi_token dist/*
            """
        }
    }
}